//! Chat with Qwen3 through your engine (greedy decoding).
//!
//!   cargo run --release --bin generate -- "Explain PagedAttention in two sentences."
//!   cargo run --release --bin generate -- --model Qwen3-1.7B --max-tokens 200 "..."
use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use minillm::config::model_dir;
use minillm::engine::{Engine, EngineConfig};
use minillm::sequence::Status;
use minillm::text::Text;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut model = "Qwen3-0.6B".to_string();
    let mut max_tokens = 128usize;
    while args.len() >= 2 && args[0].starts_with("--") {
        match args[0].as_str() {
            "--model" => model = args[1].clone(),
            "--max-tokens" => max_tokens = args[1].parse()?,
            other => anyhow::bail!("unknown flag {other}"),
        }
        args.drain(..2);
    }
    let prompt = if args.is_empty() { "What is a KV cache? Answer briefly.".to_string() } else { args.join(" ") };
    let dir = model_dir(&model);
    let text = Text::load(&dir)?;
    let t0 = Instant::now();
    let mut engine = Engine::new(&dir, EngineConfig { kv_cache_bytes: 1 << 30, ..Default::default() })?;
    eprintln!("# loaded {model} ({:.2} GB weights) in {:.1}s", engine.model.w.bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
    let ids = text.chat_prompt(&prompt)?;
    let id = engine.add_request(ids.clone(), max_tokens, text.eos.clone());
    let t1 = Instant::now();
    let mut shown = 0usize;
    let mut first: Option<f64> = None;
    while engine.seqs[&id].status != Status::Finished {
        engine.step()?;
        let out = engine.seqs[&id].output();
        if !out.is_empty() && first.is_none() {
            first = Some(t1.elapsed().as_secs_f64());
        }
        let s = text.decode(out)?;
        if s.len() > shown {
            print!("{}", &s[shown..]);
            std::io::stdout().flush()?;
            shown = s.len();
        }
    }
    let n = engine.seqs[&id].num_generated();
    let total = t1.elapsed().as_secs_f64();
    println!();
    eprintln!("# prompt {} tokens, generated {n}; TTFT {:.1} ms; decode {:.1} tok/s",
        ids.len(), first.unwrap_or(0.0) * 1e3, (n.saturating_sub(1)) as f64 / (total - first.unwrap_or(0.0)).max(1e-9));
    Ok(())
}
