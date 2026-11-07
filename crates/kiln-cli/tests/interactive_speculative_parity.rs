use std::io::Write;
use std::process::{Command, Stdio};

fn run_cli(model: &str, speculative: bool) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kiln"));
    command
        .args(["run", model, "--max-tokens", "32", "--keep-context"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if speculative {
        command.args(["--spec-type", "ngram-mod"]);
    }

    let mut child = command.spawn().expect("start KILN CLI process");
    {
        let input = child.stdin.as_mut().expect("open KILN CLI stdin");
        writeln!(input, "red blue red blue red blue red blue")
            .expect("write first prompt");
        writeln!(input, "Repeat the same two colors once.")
            .expect("write second prompt");
        writeln!(input, "/exit").expect("write exit command");
    }
    child.wait_with_output().expect("wait for KILN CLI")
}

fn responses(stdout: &[u8]) -> Vec<String> {
    let output = String::from_utf8_lossy(stdout);
    output
        .split("kiln> ")
        .skip(1)
        .map(|response| response.lines().next().unwrap_or_default().to_owned())
        .collect()
}

#[test]
fn extracts_response_lines_from_interactive_output() {
    assert_eq!(responses(b"banner\nyou> kiln> first reply\nyou> kiln> second reply\n"),
        vec!["first reply".to_owned(), "second reply".to_owned()]);
}

/// Run manually with KILN_TEST_MODEL_PATH pointing at the pinned GGUF.
/// This exercises the interactive CLI wiring, unlike the lower-level verifier test.
#[test]
#[ignore = "requires a pinned real GGUF in KILN_TEST_MODEL_PATH"]
fn interactive_ngram_matches_greedy_across_append_only_turns() {
    let model = std::env::var("KILN_TEST_MODEL_PATH")
        .expect("set KILN_TEST_MODEL_PATH to the pinned test GGUF");
    let baseline = run_cli(&model, false);
    let speculative = run_cli(&model, true);

    assert!(baseline.status.success(), "baseline failed: {}", String::from_utf8_lossy(&baseline.stderr));
    assert!(speculative.status.success(), "speculative run failed: {}", String::from_utf8_lossy(&speculative.stderr));
    assert!(String::from_utf8_lossy(&speculative.stderr).contains("[ngram-mod:"),
        "speculative run did not report its per-turn statistics");
    assert_eq!(responses(&speculative.stdout), responses(&baseline.stdout));
}
