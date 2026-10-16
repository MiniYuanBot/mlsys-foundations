//! Offline serving benchmark on a synthetic workload.
//!
//!   cargo run --release --bin bench -- [flags] > results/lab4/run.csv
//!
//! Flags (defaults in brackets):
//!   --requests N [64]   --prompt N [512]   --shared-prefix N [0]   --output N [128]
//!   --rate R [0 = all at t=0] requests/s, Poisson arrivals
//!   --max-seqs N [64]   --max-tokens N [2048]   --kv-gb F [2.0]
//!   --no-chunked   --no-prefix-cache   --static   --attention naive|fast [fast]
//!   --vary-output   output length uniform in [output/4, 7*output/4] instead of fixed
//!   --model NAME [Qwen3-0.6B]   --label TEXT (first CSV column)
//!
//! Prompts are random token ids (seeded): the first `shared-prefix` tokens are
//! identical for every request (a system prompt), the rest differ. Requests
//! never stop early (no stop tokens), so every run does the same work.
use std::time::{Duration, Instant};

use anyhow::Result;
use minillm::config::model_dir;
use minillm::engine::{Engine, EngineConfig};
use minillm::ops::AttnImpl;
use minillm::scheduler_student::SchedulerConfig;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str, def: &str| -> String {
        a.iter()
            .position(|x| x == name)
            .and_then(|i| a.get(i + 1).cloned())
            .unwrap_or_else(|| def.to_string())
    };
    let has = |name: &str| a.iter().any(|x| x == name);
    let (n_req, prompt, shared, output) = (
        get("--requests", "64").parse::<usize>()?,
        get("--prompt", "512").parse::<usize>()?,
        get("--shared-prefix", "0").parse::<usize>()?,
        get("--output", "128").parse::<usize>()?,
    );
    let rate: f64 = get("--rate", "0").parse()?;
    let cfg = EngineConfig {
        block_size: 16,
        kv_cache_bytes: (get("--kv-gb", "2.0").parse::<f64>()? * (1u64 << 30) as f64) as usize,
        sched: SchedulerConfig {
            max_num_seqs: get("--max-seqs", "64").parse()?,
            max_batched_tokens: get("--max-tokens", "2048").parse()?,
            chunked_prefill: !has("--no-chunked"),
            static_batching: has("--static"),
        },
        prefix_caching: !has("--no-prefix-cache"),
        attention: if get("--attention", "fast") == "naive" {
            AttnImpl::Naive
        } else {
            AttnImpl::Fast
        },
    };
    let label = get("--label", "run");
    let mut engine = Engine::new(&model_dir(&get("--model", "Qwen3-0.6B")), cfg.clone())?;
    eprintln!(
        "# {} KV blocks of {} tokens ({:.2} GB)",
        engine.kv.num_blocks,
        cfg.block_size,
        cfg.kv_cache_bytes as f64 / 1e9
    );

    let mut rng = Rng(0x9e3779b97f4a7c15);
    let common: Vec<u32> = (0..shared.min(prompt))
        .map(|_| 1000 + (rng.next() % 100_000) as u32)
        .collect();
    let mut arrivals = Vec::new();
    let mut t = 0.0;
    for _ in 0..n_req {
        if rate > 0.0 {
            t += -(1.0 - rng.uniform()).ln() / rate;
        }
        arrivals.push(t);
    }
    let out_lens: Vec<usize> = (0..n_req)
        .map(|_| {
            if has("--vary-output") {
                output / 4 + (rng.next() % (3 * output as u64 / 2 + 1)) as usize
            } else {
                output
            }
        })
        .collect();
    let prompts: Vec<Vec<u32>> = (0..n_req)
        .map(|_| {
            let mut p = common.clone();
            p.extend((p.len()..prompt).map(|_| 1000 + (rng.next() % 100_000) as u32));
            p
        })
        .collect();

    let start = Instant::now();
    let mut next = 0;
    let mut ids = Vec::new();
    while next < n_req || engine.has_work() {
        while next < n_req && start.elapsed() >= Duration::from_secs_f64(arrivals[next]) {
            let id = engine.add_request(prompts[next].clone(), out_lens[next].max(1), Vec::new());
            // Arrival time is the scheduled one, not when we got around to adding it.
            engine.seqs.get_mut(&id).unwrap().arrival =
                start + Duration::from_secs_f64(arrivals[next]);
            ids.push(id);
            next += 1;
        }
        if engine.has_work() {
            engine.step()?;
        } else if next < n_req {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let (mut ttft, mut tpot, mut e2e) = (Vec::new(), Vec::new(), Vec::new());
    let mut cached = 0usize;
    for id in &ids {
        let s = &engine.seqs[id];
        let first = s
            .first_token_at
            .unwrap()
            .duration_since(s.arrival)
            .as_secs_f64();
        let done = s
            .finished_at
            .unwrap()
            .duration_since(s.arrival)
            .as_secs_f64();
        ttft.push(first * 1e3);
        e2e.push(done * 1e3);
        if s.num_generated() > 1 {
            tpot.push((done - first) * 1e3 / (s.num_generated() - 1) as f64);
        }
        cached += s.num_cached_tokens;
    }
    let out_tokens: usize = ids.iter().map(|id| engine.seqs[id].num_generated()).sum();
    let st = &engine.stats;
    let kv = &engine.kv.stats;
    println!("label,requests,prompt,shared,output,rate,chunked,prefix_cache,static,attention,wall_s,output_tok_s,total_tok_s,\
ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,e2e_p50_ms,preemptions,cached_prompt_frac,prefix_hit_rate,peak_blocks,steps");
    println!("{label},{n_req},{prompt},{shared},{output},{rate},{},{},{},{:?},{wall:.3},{:.1},{:.1},{:.1},{:.1},{:.2},{:.2},{:.1},{},{:.3},{:.3},{},{}",
        cfg.sched.chunked_prefill, cfg.prefix_caching, cfg.sched.static_batching, cfg.attention,
        out_tokens as f64 / wall, (n_req * prompt + out_tokens) as f64 / wall,
        percentile(&mut ttft, 0.5), percentile(&mut ttft, 0.99), percentile(&mut tpot, 0.5), percentile(&mut tpot, 0.99),
        percentile(&mut e2e, 0.5), st.preemptions, cached as f64 / (n_req * prompt) as f64,
        kv.prefix_hits as f64 / kv.prefix_queries.max(1) as f64, st.peak_blocks_used, st.steps);
    Ok(())
}
