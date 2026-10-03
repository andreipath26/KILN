//! The Performance Monitor.
//!
//! Samples the machine every 500 milliseconds and exposes the current
//! performance envelope. See docs/core-design.md, Subsystem 1.

use std::time::{Duration, Instant, SystemTime};

use kiln_hal::ThermalState;
use serde::{Deserialize, Serialize};

/// The current performance envelope of the machine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerformanceEnvelope {
    /// Current throughput divided by cold-start baseline. 1.0 means cold.
    pub throughput_fraction: f32,
    /// Current thermal state.
    pub thermal_state: ThermalState,
    /// Bytes available in RAM before swap activity begins.
    pub ram_headroom_bytes: u64,
    /// When this sample was taken.
    #[serde(with = "serde_system_time")]
    pub sampled_at: SystemTime,
}

impl PerformanceEnvelope {
    /// A baseline envelope that assumes a cold machine at full speed.
    pub fn baseline(ram_headroom_bytes: u64) -> Self {
        Self {
            throughput_fraction: 1.0,
            thermal_state: ThermalState::Cold,
            ram_headroom_bytes,
            sampled_at: SystemTime::now(),
        }
    }
}

/// A single sample from the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub elapsed: Duration,
    pub cpu_mhz: f32,
    pub cpu_temp_c: f32,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub thermal_state: ThermalState,
}

/// Rolling thermal history over a window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalHistory {
    pub samples: Vec<Sample>,
    pub window: Duration,
}

impl ThermalHistory {
    pub fn new(window: Duration) -> Self {
        Self { samples: Vec::new(), window }
    }

    pub fn push(&mut self, sample: Sample) {
        self.samples.push(sample);
        let cutoff = sample.elapsed;
        self.samples.retain(|s| cutoff.saturating_sub(s.elapsed) <= self.window);
    }

    /// Mean CPU frequency over the window. Zero if empty.
    pub fn mean_cpu_mhz(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s.cpu_mhz).sum();
        sum / self.samples.len() as f32
    }

    /// Mean CPU temperature over the window. Zero if empty.
    pub fn mean_cpu_temp_c(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s.cpu_temp_c).sum();
        sum / self.samples.len() as f32
    }
}

/// A prediction that throttling will occur soon.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThrottlePrediction {
    /// Seconds until predicted throttling.
    pub seconds_until: u32,
    /// Confidence from 0.0 to 1.0.
    pub confidence: f32,
}

/// The monitor trait. Implemented by LinuxMonitor and MockMonitor.
pub trait PerformanceMonitor: Send + Sync {
    /// The most recent envelope.
    fn current_envelope(&self) -> PerformanceEnvelope;

    /// The rolling thermal history.
    fn thermal_history(&self, window: Duration) -> ThermalHistory;

    /// A prediction of imminent throttling, if any.
    fn predicted_throttle(&self) -> Option<ThrottlePrediction>;

    /// Record a fresh sample. Called by the monitor thread on Linux and
    /// directly by tests.
    fn sample(&mut self);
}

/// A monitor that reads /proc and /sys. Linux only.
pub struct LinuxMonitor {
    started_at: Instant,
    baseline_mhz: f32,
    latest: PerformanceEnvelope,
    history: ThermalHistory,
    ram_total_bytes: u64,
}

impl LinuxMonitor {
    /// Build a new Linux monitor. Samples once immediately to establish
    /// the baseline.
    pub fn new() -> Self {
        let ram_total = read_ram_total_bytes().unwrap_or(0);
        let mut m = Self {
            started_at: Instant::now(),
            baseline_mhz: 0.0,
            latest: PerformanceEnvelope::baseline(ram_total),
            history: ThermalHistory::new(Duration::from_secs(300)),
            ram_total_bytes: ram_total,
        };
        m.sample();
        m.baseline_mhz = m.history.mean_cpu_mhz().max(1.0);
        m
    }

    pub fn ram_total_bytes(&self) -> u64 {
        self.ram_total_bytes
    }
}

impl Default for LinuxMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor for LinuxMonitor {
    fn current_envelope(&self) -> PerformanceEnvelope {
        self.latest
    }

    fn thermal_history(&self, window: Duration) -> ThermalHistory {
        let mut h = self.history.clone();
        h.window = window;
        h
    }

    fn predicted_throttle(&self) -> Option<ThrottlePrediction> {
        // Simple linear extrapolation: if temperature has risen by more than
        // 0.5 degrees C per sample and is above 70 C, predict throttling
        // within 60 seconds.
        let samples = &self.history.samples;
        if samples.len() < 4 {
            return None;
        }
        let recent = &samples[samples.len() - 4..];
        let t0 = recent.first()?.cpu_temp_c;
        let t1 = recent.last()?.cpu_temp_c;
        let rate = (t1 - t0) / 3.0;
        if rate > 0.5 && t1 > 70.0 {
            let to_80 = (80.0 - t1) / rate;
            let seconds = to_80.max(0.0).min(60.0) as u32;
            let confidence = ((rate - 0.5) / 2.0).clamp(0.0, 1.0);
            return Some(ThrottlePrediction { seconds_until: seconds, confidence });
        }
        None
    }

    fn sample(&mut self) {
        let cpu_mhz = read_cpu_mhz().unwrap_or(0.0);
        let cpu_temp_c = read_cpu_temp_c().unwrap_or(0.0);
        let (ram_used, ram_total) = read_ram_usage().unwrap_or((0, self.ram_total_bytes));
        let thermal_state = classify_thermal(cpu_temp_c);
        let elapsed = self.started_at.elapsed();

        let sample = Sample {
            elapsed,
            cpu_mhz,
            cpu_temp_c,
            ram_used_bytes: ram_used,
            ram_total_bytes: ram_total,
            thermal_state,
        };
        self.history.push(sample);

        let throughput_fraction = if self.baseline_mhz > 0.0 {
            (cpu_mhz / self.baseline_mhz).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let ram_headroom_bytes = ram_total.saturating_sub(ram_used);

        self.latest = PerformanceEnvelope {
            throughput_fraction,
            thermal_state,
            ram_headroom_bytes,
            sampled_at: SystemTime::now(),
        };
    }
}

/// A monitor for tests. Returns whatever was last set.
pub struct MockMonitor {
    envelope: PerformanceEnvelope,
    history: ThermalHistory,
    prediction: Option<ThrottlePrediction>,
}

impl MockMonitor {
    pub fn new(envelope: PerformanceEnvelope) -> Self {
        Self {
            envelope,
            history: ThermalHistory::new(Duration::from_secs(300)),
            prediction: None,
        }
    }

    pub fn set_envelope(&mut self, envelope: PerformanceEnvelope) {
        self.envelope = envelope;
    }

    pub fn set_prediction(&mut self, p: Option<ThrottlePrediction>) {
        self.prediction = p;
    }
}

impl PerformanceMonitor for MockMonitor {
    fn current_envelope(&self) -> PerformanceEnvelope {
        self.envelope
    }
    fn thermal_history(&self, _window: Duration) -> ThermalHistory {
        self.history.clone()
    }
    fn predicted_throttle(&self) -> Option<ThrottlePrediction> {
        self.prediction
    }
    fn sample(&mut self) {}
}

// ---- Platform samplers. ----

fn read_cpu_mhz() -> Option<f32> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("cpu MHz") {
            if let Some(val) = rest.split(':').nth(1) {
                if let Ok(v) = val.trim().parse::<f32>() {
                    sum += v;
                    count += 1;
                }
            }
        }
    }
    if count == 0 { None } else { Some(sum / count as f32) }
}

fn read_cpu_temp_c() -> Option<f32> {
    let mut hottest = 0.0f32;
    let entries = std::fs::read_dir("/sys/class/thermal").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("thermal_zone") {
            continue;
        }
        let temp_path = entry.path().join("temp");
        if let Ok(s) = std::fs::read_to_string(&temp_path) {
            if let Ok(v) = s.trim().parse::<f32>() {
                let c = v / 1000.0;
                if c > hottest {
                    hottest = c;
                }
            }
        }
    }
    if hottest > 0.0 { Some(hottest) } else { None }
}

fn read_ram_total_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb = rest.trim().split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn read_ram_usage() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = 0u64;
    let mut available = 0u64;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = rest.trim().split_whitespace().next()?.parse::<u64>().ok()? * 1024;
        }
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = rest.trim().split_whitespace().next()?.parse::<u64>().ok()? * 1024;
        }
    }
    if total == 0 { None } else { Some((total.saturating_sub(available), total)) }
}

fn classify_thermal(temp_c: f32) -> ThermalState {
    if temp_c < 60.0 {
        ThermalState::Cold
    } else if temp_c < 80.0 {
        ThermalState::Warm
    } else if temp_c < 90.0 {
        ThermalState::Throttling
    } else {
        ThermalState::SeverelyThrottling
    }
}

mod serde_system_time {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let d = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        d.as_secs().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}
