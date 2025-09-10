"""Repository rule for SYCL autoconfiguration.
`sycl_configure` depends on the following environment variables:
  * `TF_NEED_SYCL`: Whether to enable building with SYCL.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
"""

load("//third_party/gpus/sycl:level_zero.bzl", "level_zero_redist")
load("//third_party/gpus/sycl:sycl_dl_essential.bzl", "sycl_redist")
load(
    "//third_party/remote_config:common.bzl",
    "err_out",
    "execute",
    "files_exist",
    "get_bash_bin",
    "get_host_environ",
    "get_python_bin",
    "raw_exec",
    "realpath",
    "which",
)
load(
    ":compiler_common_tools.bzl",
    "to_list_of_strings",
)
load(
    ":cuda_configure.bzl",
    "make_copy_dir_rule",
    "make_copy_files_rule",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_CLANG_HOST_COMPILER_PATH = "CLANG_COMPILER_PATH"
_CLANG_HOST_COMPILER_PREFIX = "CLANG_HOST_COMPILER_PATH"

def _mkl_include_path(sycl_config):
    return sycl_config.mkl_include_dir

def _mkl_library_path(sycl_config):
    return sycl_config.mkl_library_dir

def _l0_include_path(sycl_config):
    return sycl_config.l0_include_dir

def _l0_library_path(sycl_config):
    return sycl_config.l0_library_dir

def _repo_root(repository_ctx, repo_name):
    # Resolve external repo root in a CUDA-like way.
    # Ensure @sycl_hermetic has a root BUILD (build_file_content in WORKSPACE).
    return str(repository_ctx.path(Label("%s//:BUILD" % repo_name)).dirname)

def _first_existing(ctx, paths):
    for p in paths:
        if ctx.path(p).exists:
            return p
    return None

def _sycl_header_path(repository_ctx, sycl_config, bash_bin):
    # Only accept <toolkit>/include (bundle does not use linux/include for headers here).
    sycl_header_path = sycl_config.sycl_toolkit_path
    include_dir = sycl_header_path + "/include"
    if not files_exist(repository_ctx, [include_dir], bash_bin)[0]:
        auto_configure_fail("Cannot find SYCL headers at %s" % include_dir)
    return sycl_header_path

def _sycl_include_path(repository_ctx, sycl_config, bash_bin):
    """Return cxx_builtin_include_directory entries for SYCL + MKL."""
    inc_dirs = []
    inc_dirs.append(_mkl_include_path(sycl_config))
    inc_dirs.append(_sycl_header_path(repository_ctx, sycl_config, bash_bin) + "/include")
    inc_dirs.append(_sycl_header_path(repository_ctx, sycl_config, bash_bin) + "/include/sycl")
    return inc_dirs

def enable_sycl(repository_ctx):
    return bool(get_host_environ(repository_ctx, "TF_NEED_SYCL", "").strip())

def _use_icpx_and_clang(repository_ctx):
    return get_host_environ(repository_ctx, "TF_ICPX_CLANG", "").strip()

def auto_configure_fail(msg):
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def find_cc(repository_ctx):
    # Pick host C/C++ compiler path (for builtin include discovery)
    if _use_icpx_and_clang(repository_ctx):
        target_cc_name = "clang"
        cc_path_envvar = _CLANG_HOST_COMPILER_PATH
    else:
        target_cc_name = "gcc"
        cc_path_envvar = _GCC_HOST_COMPILER_PATH
    cc_name = get_host_environ(repository_ctx, cc_path_envvar) or target_cc_name
    if cc_name.startswith("/"):
        return cc_name
    cc = which(repository_ctx, cc_name)
    if cc == None:
        fail(("Cannot find {}, either correct your path or set the {}" +
              " environment variable").format(target_cc_name, cc_path_envvar))
    return cc

def find_sycl_root(repository_ctx, sycl_config):
    sycl_name = str(repository_ctx.path(sycl_config.sycl_toolkit_path.strip()).realpath)
    if sycl_name.startswith("/"):
        return sycl_name
    fail("Cannot find DPC++ compiler, please correct your path")

def find_sycl_include_path(repository_ctx, sycl_config):
    # Discover default compiler include directories (clang/gcc resource dirs)
    base_path = find_sycl_root(repository_ctx, sycl_config)
    bin_path = repository_ctx.path(base_path + "/bin/icpx")
    icpx_extra = ""
    if not bin_path.exists:
        bin_path = repository_ctx.path(base_path + "/bin/compiler/clang")
        if not bin_path.exists:
            fail("Cannot find DPC++ compiler, please correct your path")
    else:
        icpx_extra = "-fsycl"
    if _use_icpx_and_clang(repository_ctx):
        clang_path = repository_ctx.which("clang")
        clang_install_dir = repository_ctx.execute([clang_path, "-print-resource-dir"])
        clang_install_dir_opt = "--sysroot=" + str(repository_ctx.path(clang_install_dir.stdout.strip()).dirname)
        cmd_out = repository_ctx.execute([
            bin_path, icpx_extra, clang_install_dir_opt,
            "-xc++", "-E", "-v", "/dev/null", "-o", "/dev/null",
        ])
    else:
        gcc_path = repository_ctx.which("gcc")
        gcc_install_dir = repository_ctx.execute([gcc_path, "-print-libgcc-file-name"])
        gcc_install_dir_opt = "--gcc-install-dir=" + str(repository_ctx.path(gcc_install_dir.stdout.strip()).dirname)
        cmd_out = repository_ctx.execute([
            bin_path, icpx_extra, gcc_install_dir_opt,
            "-xc++", "-E", "-v", "/dev/null", "-o", "/dev/null",
        ])

    outlist = cmd_out.stderr.split("\n")
    include_dirs = []
    for l in outlist:
        if l.startswith(" ") and l.strip().startswith("/"):
            realp = str(repository_ctx.path(l.strip()).realpath)
            if realp not in include_dirs:
                include_dirs.append(realp)
    return include_dirs

def _lib_name(lib, version = "", static = False):
    if static:
        return "lib%s.a" % lib
    else:
        if version:
            version = ".%s" % version
        return "lib%s.so%s" % (lib, version)

def _sycl_lib_paths(repository_ctx, lib, basedir):
    file_name = _lib_name(lib, version = "", static = False)
    return [repository_ctx.path("%s/%s" % (basedir, file_name))]

def _batch_files_exist(repository_ctx, libs_paths, bash_bin):
    all_paths = []
    for _, lib_paths in libs_paths:
        for lib_path in lib_paths:
            all_paths.append(lib_path)
    return files_exist(repository_ctx, all_paths, bash_bin)

def _select_sycl_lib_paths(repository_ctx, libs_paths, bash_bin):
    test_results = _batch_files_exist(repository_ctx, libs_paths, bash_bin)
    libs = {}
    i = 0
    for name, lib_paths in libs_paths:
        selected_path = None
        for path in lib_paths:
            if test_results[i] and selected_path == None:
                selected_path = path
            i = i + 1
        if selected_path == None:
            auto_configure_fail("Cannot find sycl library %s in %s" % (name, path))
        libs[name] = struct(
            file_name = selected_path.basename,
            path = realpath(repository_ctx, selected_path, bash_bin),
        )
    return libs

def _find_libs(repository_ctx, sycl_config, bash_bin):
    mkl_path = _mkl_library_path(sycl_config)
    libs_paths = [
        (name, _sycl_lib_paths(repository_ctx, name, mkl_path))
        for name in ["mkl_intel_ilp64", "mkl_sequential", "mkl_core"]
    ]
    if sycl_config.sycl_basekit_version_number < "2024":
        libs_paths.append(("mkl_sycl", _sycl_lib_paths(repository_ctx, "mkl_sycl", mkl_path)))
    else:
        for name in [
            "mkl_sycl_blas", "mkl_sycl_lapack", "mkl_sycl_sparse",
            "mkl_sycl_dft", "mkl_sycl_vm", "mkl_sycl_rng",
            "mkl_sycl_stats", "mkl_sycl_data_fitting",
        ]:
            libs_paths.append((name, _sycl_lib_paths(repository_ctx, name, mkl_path)))

    # Level Zero is OPTIONAL in hermetic mode: add ze_loader only if a lib dir exists.
    l0_path = _l0_library_path(sycl_config)
    if l0_path:
        libs_paths.append(("ze_loader", _sycl_lib_paths(repository_ctx, "ze_loader", l0_path)))

    return _select_sycl_lib_paths(repository_ctx, libs_paths, bash_bin)

def find_sycl_config(repository_ctx):
    python_bin = get_python_bin(repository_ctx)
    exec_result = execute(repository_ctx, [python_bin, repository_ctx.attr._find_sycl_config])
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_sycl_config.py: %s" % err_out(exec_result))
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _get_sycl_config(repository_ctx, bash_bin):
    config = find_sycl_config(repository_ctx)
    return struct(
        sycl_basekit_path = config["sycl_basekit_path"],
        sycl_toolkit_path = config["sycl_toolkit_path"],
        sycl_version_number = config["sycl_version_number"],
        sycl_basekit_version_number = config["sycl_basekit_version_number"],
        mkl_include_dir = config["mkl_include_dir"],
        mkl_library_dir = config["mkl_library_dir"],
        l0_include_dir = config["l0_include_dir"],
        l0_library_dir = config["l0_library_dir"],
    )

def _tpl_path(repository_ctx, labelname):
    return repository_ctx.path(Label("//third_party/gpus/%s.tpl" % labelname))

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(out, _tpl_path(repository_ctx, tpl), substitutions)

_INC_DIR_MARKER_BEGIN = "#include <...>"

def _cxx_inc_convert(path):
    return path.strip()

def _normalize_include_path(repository_ctx, path):
    path = str(repository_ctx.path(path))
    crosstool_folder = str(repository_ctx.path(".").get_child("crosstool"))
    if path.startswith(crosstool_folder):
        return "\"" + path[len(crosstool_folder) + 1:] + "\""
    return "\"" + path + "\""

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    lang = "c++" if lang_is_cpp else "c"
    result = raw_exec(repository_ctx, [cc, "-no-canonical-prefixes", "-E", "-x" + lang, "-", "-v"])
    stderr = err_out(result)
    index1 = stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = stderr[index1 + 1:]
    else:
        inc_dirs = stderr[index1 + 1:index2].strip()
    return [str(repository_ctx.path(_cxx_inc_convert(p))) for p in inc_dirs.split("\n")]

def get_cxx_inc_directories(repository_ctx, cc):
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)
    includes_cpp_set = depset(includes_cpp)
    return includes_cpp + [inc for inc in includes_c if inc not in includes_cpp_set.to_list()]

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=sycl but TensorFlow is not configured " +
       "to build with GPU support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with GPU support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")

error_gpu_disabled()
"""

def _create_dummy_repository(
        repository_ctx,
        sycl_libs = None,
        mkl_sycl_libs = None,
        copy_rules = None,
        level_zero_libs = None,
        level_zero_headers = None):
    # Minimal repo that errors clearly when --config=sycl is used but SYCL is not configured.
    sycl_libs = sycl_libs or []
    mkl_sycl_libs = mkl_sycl_libs or []
    copy_rules = copy_rules or []
    level_zero_libs = level_zero_libs or []
    level_zero_headers = level_zero_headers or []

    repository_ctx.file("crosstool/error_gpu_disabled.bzl", _DUMMY_CROSSTOOL_BZL_FILE)
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

    _tpl(
        repository_ctx,
        "sycl:build_defs.bzl",
        {"%{sycl_is_configured}": "False", "%{sycl_build_is_configured}": "False"},
    )

    _tpl(
        repository_ctx,
        "sycl:BUILD",
        {
            "%{mkl_intel_ilp64_src}": "",
            "%{mkl_sequential_src}": "",
            "%{mkl_core_src}": "",
            "%{mkl_sycl_srcs}": "",
            "%{mkl_intel_ilp64_lib}": "",
            "%{mkl_sequential_lib}": "",
            "%{mkl_core_lib}": "",
            "%{mkl_sycl_libs}": "",
            "%{level_zero_libs}": "",
            "%{level_zero_headers}": "",
            "%{sycl_headers}": "",
            "%{copy_rules}": "\n".join(copy_rules) if copy_rules else "",
        },
    )

def _create_local_sycl_repository(repository_ctx):
    tpl_paths = {labelname: _tpl_path(repository_ctx, labelname) for labelname in [
        "sycl:build_defs.bzl",
        "sycl:BUILD",
        "crosstool:BUILD.sycl",
        "crosstool:sycl_cc_toolchain_config.bzl",
        "crosstool:clang/bin/crosstool_wrapper_driver_sycl",
        "crosstool:clang/bin/ar_driver_sycl",
    ]}

    bash_bin = get_bash_bin(repository_ctx)

    hermetic = get_host_environ(repository_ctx, "SYCL_BUILD_HERMETIC", "") == "1"
    if hermetic:
        install_path = _repo_root(repository_ctx, "@sycl_hermetic")
        oneapi_version = get_host_environ(repository_ctx, "ONEAPI_VERSION", "2025.1").strip() or "2025.1"

        # Allow both archive layouts: with or without top-level 'oneapi/'.
        base = install_path + "/oneapi" if repository_ctx.path(install_path + "/oneapi").exists else install_path

        # SYCL toolkit & MKL paths (prefer lib/intel64 if present)
        sycl_toolkit_path = base + "/compiler/" + oneapi_version
        mkl_include_dir  = base + "/mkl/" + oneapi_version + "/include"
        mkl_library_dir  = _first_existing(repository_ctx, [
            base + "/mkl/" + oneapi_version + "/lib/intel64",
            base + "/mkl/" + oneapi_version + "/lib",
        ])

        if not (repository_ctx.path(sycl_toolkit_path + "/include").exists):
            auto_configure_fail("Cannot find SYCL headers at %s/include" % sycl_toolkit_path)
        if not (repository_ctx.path(mkl_include_dir).exists and mkl_library_dir):
            auto_configure_fail("Missing MKL include/lib under %s" % base)

        # Level Zero is optional: use compiler tree if present, else leave empty strings.
        l0_include_dir = _first_existing(repository_ctx, [
            sycl_toolkit_path + "/include/level_zero",
            sycl_toolkit_path + "/linux/include/level_zero",
        ]) or ""
        l0_library_dir = _first_existing(repository_ctx, [
            sycl_toolkit_path + "/lib",
            sycl_toolkit_path + "/linux/lib",
        ]) or ""

        sycl_config = struct(
            sycl_basekit_path = base + "/",  # keep trailing slash for compatibility
            sycl_toolkit_path = sycl_toolkit_path,
            sycl_version_number = "80000",
            sycl_basekit_version_number = oneapi_version,
            mkl_include_dir = mkl_include_dir,
            mkl_library_dir = mkl_library_dir,
            l0_include_dir = l0_include_dir,
            l0_library_dir = l0_library_dir,
        )
    else:
        # Non-hermetic: detect oneAPI on the host
        install_path = get_host_environ(repository_ctx, "SYCL_TOOLKIT_PATH", "") or "/opt/intel/oneapi/compiler/2025.1"
        repository_ctx.report_progress("Falling back to default SYCL path: %s" % install_path)
        sycl_config = _get_sycl_config(repository_ctx, bash_bin)

    # ---- Copy headers/libs into execroot ----
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "sycl-include",
            src_dir = _sycl_header_path(repository_ctx, sycl_config, bash_bin) + "/include",
            out_dir = "sycl/include",
        ),
    ]
    copy_rules.append(make_copy_dir_rule(
        repository_ctx,
        name = "mkl-include",
        src_dir = _mkl_include_path(sycl_config),
        out_dir = "sycl/include",
    ))

    # Only copy Level Zero headers if available (optional)
    if _l0_include_path(sycl_config):
        copy_rules.append(make_copy_dir_rule(
            repository_ctx,
            name = "level-zero-include",
            src_dir = _l0_include_path(sycl_config),
            out_dir = "level_zero/include/level_zero",
        ))

    sycl_libs = _find_libs(repository_ctx, sycl_config, bash_bin)
    sycl_lib_srcs = []
    sycl_lib_outs = []
    for lib in sycl_libs.values():
        sycl_lib_srcs.append(lib.path)
        sycl_lib_outs.append("sycl/lib/" + lib.file_name)
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "sycl-lib",
        srcs = sycl_lib_srcs,
        outs = sycl_lib_outs,
    ))

    # ---- sycl/ BUILD + defs ----
    repository_ctx.template(
        "sycl/build_defs.bzl",
        tpl_paths["sycl:build_defs.bzl"],
        {"%{sycl_is_configured}": "True", "%{sycl_build_is_configured}": "True"},
    )

    if sycl_config.sycl_basekit_version_number < "2024":
        mkl_sycl_libs = '"%s"' % ("sycl/lib/" + sycl_libs["mkl_sycl"].file_name)
    else:
        mkl_sycl_libs = '"%s",\n"%s",\n"%s",\n"%s",\n"%s",\n"%s",\n"%s",\n"%s",\n' % (
            "sycl/lib/" + sycl_libs["mkl_sycl_blas"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_lapack"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_sparse"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_dft"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_vm"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_rng"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_stats"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_data_fitting"].file_name,
        )

    # Only reference ze_loader if we actually found it
    level_zero_libs = ""
    if "ze_loader" in sycl_libs:
        level_zero_libs = '"%s",\n' % ("sycl/lib/" + sycl_libs["ze_loader"].file_name)

    def _fmt_src(path):
        return '"%s",\n' % path

    repository_ctx.template(
        "sycl/BUILD",
        tpl_paths["sycl:BUILD"],
        {
            "%{mkl_intel_ilp64_src}": _fmt_src("sycl/lib/" + sycl_libs["mkl_intel_ilp64"].file_name),
            "%{mkl_sequential_src}": _fmt_src("sycl/lib/" + sycl_libs["mkl_sequential"].file_name),
            "%{mkl_core_src}":      _fmt_src("sycl/lib/" + sycl_libs["mkl_core"].file_name),
            "%{mkl_sycl_srcs}":     mkl_sycl_libs,
            "%{mkl_intel_ilp64_lib}": sycl_libs["mkl_intel_ilp64"].file_name,
            "%{mkl_sequential_lib}":  sycl_libs["mkl_sequential"].file_name,
            "%{mkl_core_lib}":        sycl_libs["mkl_core"].file_name,
            "%{mkl_sycl_libs}":       mkl_sycl_libs,
            "%{copy_rules}":          "\n".join(copy_rules),
            "%{sycl_headers}":        '":mkl-include",\n":sycl-include",\n',
            "%{level_zero_libs}":     level_zero_libs,
            "%{level_zero_headers}":  ('":level-zero-include"' if _l0_include_path(sycl_config) else ""),
        },
    )

    # ---- crosstool/ config ----
    is_icpx_and_clang = _use_icpx_and_clang(repository_ctx)
    cc = find_cc(repository_ctx)

    host_compiler_includes = get_cxx_inc_directories(repository_ctx, cc)
    clang_host_compiler_prefix = get_host_environ(repository_ctx, _CLANG_HOST_COMPILER_PREFIX, "/usr/bin")
    gcc_host_compiler_prefix   = get_host_environ(repository_ctx, _GCC_HOST_COMPILER_PREFIX, "/usr/bin")

    sycl_defines = {}
    sycl_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_sycl"
    if is_icpx_and_clang:
        sycl_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-no-canonical-prefixes\""
        sycl_defines["%{host_compiler_prefix}"] = clang_host_compiler_prefix
    else:
        sycl_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""
        sycl_defines["%{host_compiler_prefix}"] = gcc_host_compiler_prefix

    sycl_defines["%{ar_path}"] = "clang/bin/ar_driver_sycl"
    sycl_defines["%{cpu_compiler}"] = str(cc)
    sycl_defines["%{linker_bin_path}"] = "/usr/bin"

    sycl_internal_inc_dirs = find_sycl_include_path(repository_ctx, sycl_config)
    cxx_builtin_includes_list = sycl_internal_inc_dirs + _sycl_include_path(repository_ctx, sycl_config, bash_bin) + host_compiler_includes
    sycl_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(cxx_builtin_includes_list)
    sycl_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_SYCL=1",
        "-DMKL_ILP64",
        "-fPIC",
    ])
    sycl_defines["%{sycl_compiler_root}"] = str(sycl_config.sycl_toolkit_path)
    sycl_defines["%{SYCL_ROOT_DIR}"]     = str(sycl_config.sycl_toolkit_path)
    sycl_defines["%{basekit_path}"]      = str(sycl_config.sycl_basekit_path)
    sycl_defines["%{basekit_version}"]   = str(sycl_config.sycl_basekit_version_number)

    repository_ctx.template("crosstool/BUILD",                 tpl_paths["crosstool:BUILD.sycl"],                sycl_defines)
    repository_ctx.template("crosstool/cc_toolchain_config.bzl", tpl_paths["crosstool:sycl_cc_toolchain_config.bzl"], sycl_defines)
    repository_ctx.template("crosstool/clang/bin/crosstool_wrapper_driver_sycl", tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_sycl"], sycl_defines)
    repository_ctx.template("crosstool/clang/bin/ar_driver_sycl",                tpl_paths["crosstool:clang/bin/ar_driver_sycl"],                sycl_defines)

def _sycl_autoconf_imp(repository_ctx):
    if not enable_sycl(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        _create_local_sycl_repository(repository_ctx)

sycl_configure = repository_rule(
    implementation = _sycl_autoconf_imp,
    local = True,
    attrs = {
        "_find_sycl_config": attr.label(default = Label("//third_party/gpus:find_sycl_config.py")),
    },
)
