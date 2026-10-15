//! Speculative decoding benchmark: Qwen3-1.7B target, Qwen3-0.6B draft.
//!
//!   cargo run --release --bin bench -- [--tokens 192] [--prompts all|chat|code|copy]
//!
//! One CSV row per (prompt, method): tokens/s, speedup over plain decoding,
//! acceptance rate, tokens per target forward, forward-call counts, and how
//! many leading tokens equal plain greedy decoding.
use anyhow::Result;
use minillm::config::model_dir;
use minillm::model_student::Qwen3;
use minillm::text::Text;
use specdec::decode::{generate, Method};
use specdec::ops5;

const CODE: &str = r#"fn binary_search(xs: &[i64], target: i64) -> Option<usize> {
    let (mut lo, mut hi) = (0usize, xs.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if xs[mid] == target { return Some(mid); }
        if xs[mid] < target { lo = mid + 1; } else { hi = mid; }
    }
    None
}
"#;

const COPY: &str = "The KV cache stores keys and values of past tokens so that each decoding step only computes attention for the new token. ";

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str, d: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1).cloned())
            .unwrap_or_else(|| d.into())
    };
    let tokens: usize = get("--tokens", "192").parse()?;
    let which = get("--prompts", "all");
    let gpu = ops5::gpu()?;
    let target = Qwen3::load(&gpu, &model_dir("Qwen3-1.7B"))?;
    let draft = Qwen3::load(&gpu, &model_dir("Qwen3-0.6B"))?;
    let text = Text::load(&model_dir("Qwen3-1.7B"))?;
    let prompts: Vec<(&str, Vec<u32>)> = vec![
        (
            "chat",
            text.chat_prompt(
                "Explain, step by step, how a CPU cache works and why it matters for performance.",
            )?,
        ),
        (
            "code",
            text.chat_prompt(&format!(
                "Add doc comments to this Rust function, keeping the code identical:\n\n{CODE}"
            ))?,
        ),
        (
            "copy",
            text.chat_prompt(&format!(
                "Repeat the following text exactly, then say DONE.\n\n{}",
                COPY.repeat(4)
            ))?,
        ),
    ];
    let methods: Vec<(&str, Method)> = vec![
        ("plain", Method::Plain),
        ("ngram_k4", Method::Ngram { n: 3, k: 4 }),
        ("ngram_k8", Method::Ngram { n: 3, k: 8 }),
        ("draft_k2", Method::Draft { k: 2 }),
        ("draft_k3", Method::Draft { k: 3 }),
        ("draft_k5", Method::Draft { k: 5 }),
        (
            "tree_221",
            Method::Tree {
                widths: vec![2, 2, 1],
            },
        ),
        (
            "tree_3211",
            Method::Tree {
                widths: vec![3, 2, 1, 1],
            },
        ),
        (
            "tree_4221",
            Method::Tree {
                widths: vec![4, 2, 2, 1],
            },
        ),
        (
            "sample_T0.8_k3",
            Method::Sample {
                k: 3,
                temp: 0.8,
                seed: 1,
            },
        ),
    ];
    println!("prompt,method,tokens,secs,tok_s,speedup,acceptance,tokens_per_target_call,target_calls,draft_calls,same_as_plain");
    for (pname, prompt) in &prompts {
        if which != "all" && which != *pname {
            continue;
        }
        let mut base: Option<(f64, Vec<u32>)> = None;
        for (mname, m) in &methods {
            let r = match generate(&gpu, &target, Some(&draft), prompt, tokens, m) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("# skip {pname}/{mname}: {e}");
                    continue;
                }
            };
            let (b_tps, b_out) = base
                .get_or_insert_with(|| (r.tokens_per_sec(), r.output.clone()))
                .clone();
            let same = r
                .output
                .iter()
                .zip(&b_out)
                .take_while(|(x, y)| x == y)
                .count();
            println!(
                "{pname},{mname},{},{:.3},{:.1},{:.2},{:.3},{:.2},{},{},{same}",
                r.output.len(),
                r.secs,
                r.tokens_per_sec(),
                r.tokens_per_sec() / b_tps,
                r.acceptance_rate(),
                r.output.len() as f64 / r.target_calls.max(1) as f64,
                r.target_calls,
                r.draft_calls
            );
        }
    }
    Ok(())
}
