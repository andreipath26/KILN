use kiln_core::{LinuxMonitor, PerformanceMonitor};
use std::time::Duration;

fn main() {
    let mut m = LinuxMonitor::new();
    m.sample();
    let env = m.current_envelope();
    let hist = m.thermal_history(Duration::from_secs(60));
    let pred = m.predicted_throttle();

    println!("=== KILN MONITOR TEST ===");
    println!("throughput_fraction: {:.3}", env.throughput_fraction);
    println!("thermal_state:       {:?}", env.thermal_state);
    println!("ram_headroom_bytes:  {} ({:.2} GB)", env.ram_headroom_bytes, env.ram_headroom_bytes as f64 / 1_073_741_824.0);
    println!("ram_total_bytes:     {} ({:.2} GB)", m.ram_total_bytes(), m.ram_total_bytes() as f64 / 1_073_741_824.0);
    println!("history_samples:     {}", hist.samples.len());
    println!("mean_cpu_mhz:        {:.1}", hist.mean_cpu_mhz());
    println!("mean_cpu_temp_c:     {:.1}", hist.mean_cpu_temp_c());
    println!("predicted_throttle:  {:?}", pred);
    println!("=== END ===");
}
