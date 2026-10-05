//! System profile. Probed at first run, read on every run.
//! See docs/system-profile-design.md.
//!
//! No external crates. Core count from std. RAM from /proc/meminfo
//! on Linux, sysctl on macOS, GlobalMemoryStatusEx on Windows. The
//! first version targets Linux only; the other platforms return
//! defaults and are flagged in the profile.

use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SystemProfile {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub cpu_cores: u32,
    pub ram_gb: u32,
    pub tier: Tier,
    pub probed_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tier {
    Tier0,
    Tier1,
    Workstation,
    Server,
}

impl SystemProfile {
    /// Probe the current machine and return a fresh profile.
    pub fn probe() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);
        let ram_gb = read_ram_gb().unwrap_or(8);
        let tier = classify(cores, ram_gb);
        let (n_ctx, n_batch, n_threads) = params_for(tier, cores);
        Self {
            n_ctx,
            n_batch,
            n_threads,
            cpu_cores: cores,
            ram_gb,
            tier,
            probed_at: now_iso8601(),
        }
    }

    /// Read the profile from disk, or probe and write if absent.
    pub fn load_or_probe() -> Self {
        let path = profile_path();
        if let Ok(text) = fs::read_to_string(&path) {
            if let Some(p) = parse(&text) {
                return p;
            }
        }
        let p = Self::probe();
        let _ = p.save();
        p
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = profile_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = format!(
            "{{\n  \"n_ctx\": {},\n  \"n_batch\": {},\n  \"n_threads\": {},\n  \"cpu_cores\": {},\n  \"ram_gb\": {},\n  \"tier\": \"{:?}\",\n  \"probed_at\": \"{}\"\n}}\n",
            self.n_ctx, self.n_batch, self.n_threads,
            self.cpu_cores, self.ram_gb, self.tier, self.probed_at
        );
        fs::write(&path, text)
    }
}

pub fn profile_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg).join("kiln").join("system-profile.json");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".config").join("kiln").join("system-profile.json");
    }
    PathBuf::from("/tmp/kiln-system-profile.json")
}

fn read_ram_gb() -> Option<u32> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some((kb / 1024 / 1024) as u32);
        }
    }
    None
}

fn classify(cores: u32, ram_gb: u32) -> Tier {
    // cores is logical (includes hyperthreads). ram_gb is MemTotal.
    if cores <= 8 && ram_gb <= 16 { Tier::Tier0 }
    else if cores <= 16 && ram_gb <= 32 { Tier::Tier1 }
    else if ram_gb <= 128 { Tier::Workstation }
    else { Tier::Server }
}

fn params_for(tier: Tier, cores: u32) -> (u32, u32, u32) {
    // n_threads caps at physical cores where known. Hyperthreads do
    // not help a memory-bound matmul; they contend for the same
    // memory bus. On Tier0 that is 4.
    match tier {
        Tier::Tier0 => (4096, 512, cores.min(4)),
        Tier::Tier1 => (8192, 1024, (cores / 2).max(4)),
        Tier::Workstation => (32768, 2048, cores),
        Tier::Server => (65536, 4096, cores),
    }
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{}", secs)
}

fn parse(text: &str) -> Option<SystemProfile> {
    // Minimal parser. One value per line, "key": value,.
    let mut p = SystemProfile::probe();
    let mut found = 0;
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        if !line.contains(':') { continue; }
        let mut parts = line.splitn(2, ':');
        let k = parts.next()?.trim().trim_matches('"').trim_matches('{').trim_matches('}');
        let v = parts.next()?.trim().trim_matches('"').trim_matches('{').trim_matches('}');
        match k {
            "n_ctx" => { p.n_ctx = v.parse().ok()?; found += 1; }
            "n_batch" => { p.n_batch = v.parse().ok()?; found += 1; }
            "n_threads" => { p.n_threads = v.parse().ok()?; found += 1; }
            "cpu_cores" => { p.cpu_cores = v.parse().ok()?; found += 1; }
            "ram_gb" => { p.ram_gb = v.parse().ok()?; found += 1; }
            "probed_at" => { p.probed_at = v.to_string(); found += 1; }
            _ => {}
        }
    }
    if found >= 4 { Some(p) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_sane_values() {
        let p = SystemProfile::probe();
        assert!(p.cpu_cores >= 1);
        assert!(p.n_ctx >= 1024);
        assert!(p.n_threads >= 1);
        assert!(p.n_threads <= p.cpu_cores);
    }

    #[test]
    fn tier0_params() {
        let (ctx, batch, threads) = params_for(Tier::Tier0, 4);
        assert_eq!(ctx, 4096);
        assert_eq!(batch, 512);
        assert_eq!(threads, 4);
    }

    #[test]
    fn roundtrip() {
        let p = SystemProfile::probe();
        let text = format!(
            "{{\n  \"n_ctx\": {},\n  \"n_batch\": {},\n  \"n_threads\": {},\n  \"cpu_cores\": {},\n  \"ram_gb\": {},\n  \"probed_at\": \"{}\"\n}}\n",
            p.n_ctx, p.n_batch, p.n_threads, p.cpu_cores, p.ram_gb, p.probed_at
        );
        let q = parse(&text).expect("parse failed");
        assert_eq!(p.n_ctx, q.n_ctx);
        assert_eq!(p.n_batch, q.n_batch);
        assert_eq!(p.n_threads, q.n_threads);
    }
}
