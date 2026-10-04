//! Runtime loading of llama.cpp. See docs/runtime-loading-design.md.

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum LoadError {
    NotFound(PathBuf),
    OpenFailed { path: PathBuf, msg: String },
    MissingSymbol { name: String, msg: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::NotFound(p) => write!(f, "library not found: {}", p.display()),
            LoadError::OpenFailed { path, msg } => {
                write!(f, "failed to open {}: {}", path.display(), msg)
            }
            LoadError::MissingSymbol { name, msg } => {
                write!(f, "missing symbol {}: {}", name, msg)
            }
        }
    }
}

impl std::error::Error for LoadError {}

pub fn library_filename() -> &'static str {
    #[cfg(target_os = "linux")]   { "libllama.so" }
    #[cfg(target_os = "macos")]   { "libllama.dylib" }
    #[cfg(target_os = "windows")] { "llama.dll" }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    { "libllama.so" }
}

pub fn search_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("KILN_LLAMA_LIB") {
        out.push(PathBuf::from(p));
    }
    #[cfg(target_os = "linux")]
    {
        out.push(PathBuf::from("/usr/lib").join(library_filename()));
        out.push(PathBuf::from("/usr/local/lib").join(library_filename()));
        if let Some(home) = std::env::var_os("HOME") {
            out.push(PathBuf::from(home).join(".local/lib").join(library_filename()));
        }
    }
    if let Some(dir) = kiln_data_dir() {
        out.push(dir.join("llama.cpp").join(library_filename()));
    }
    out.push(PathBuf::from("/home/andreipath/llama.cpp/build/bin").join(library_filename()));
    out
}

pub fn kiln_data_dir() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    { std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share/kiln")) }
    #[cfg(target_os = "macos")]
    { std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/KILN")) }
    #[cfg(target_os = "windows")]
    { std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join("KILN")) }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    { None }
}

pub fn find_existing() -> Option<PathBuf> {
    for p in search_paths() {
        if p.exists() { return Some(p); }
    }
    None
}

pub struct Library {
    #[cfg(feature = "dynamic-llama")]
    inner: libloading::Library,
    path: PathBuf,
}

impl Library {
    pub fn open(path: &Path) -> Result<Self, LoadError> {
        #[cfg(feature = "dynamic-llama")]
        {
            let inner = unsafe { libloading::Library::new(path) }
                .map_err(|e| LoadError::OpenFailed {
                    path: path.to_path_buf(),
                    msg: e.to_string(),
                })?;
            Ok(Self { inner, path: path.to_path_buf() })
        }
        #[cfg(not(feature = "dynamic-llama"))]
        {
            Ok(Self { path: path.to_path_buf() })
        }
    }

    pub fn path(&self) -> &Path { &self.path }

    #[cfg(feature = "dynamic-llama")]
    pub unsafe fn symbol(&self, name: &str) -> Result<*mut std::os::raw::c_void, LoadError> {
        let sym: libloading::Symbol<*mut std::os::raw::c_void> =
            self.inner.get(name.as_bytes()).map_err(|e| LoadError::MissingSymbol {
                name: name.to_string(),
                msg: e.to_string(),
            })?;
        Ok(*sym)
    }
}
