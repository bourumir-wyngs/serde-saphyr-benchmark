//! Compare JSON parsing throughput using Criterion.
//! Benchmarks serde_json on a large synthetic JSON document.

// --- Optional: fix allocator for more stable numbers ---
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use serde::Deserialize;

// --------------------- Types under test ---------------------

#[derive(Debug, Deserialize)]
struct Document {
    defaults: Defaults,
    items: Vec<Item>,
}

#[derive(Debug, Deserialize)]
struct Defaults {
    enabled: bool,
    roles: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Item {
    enabled: bool,
    roles: Vec<String>,
    id: usize,
    name: String,
    details: Details,
}

#[derive(Debug, Deserialize)]
struct Details {
    description: String,
    notes: Vec<String>,
}

// --------------------- Data generation ---------------------

/// Build a large JSON string roughly `target_size` bytes long.
///
/// The structure mirrors the YAML benchmark to keep parsing work comparable.
fn build_large_json(target_size: usize, notes_per_item: usize) -> String {
    // We append items until the size target is reached.
    // Pre-allocate to reduce reallocations which could affect timings when cloning the String.
    let mut s = String::with_capacity(target_size + 1024);
    s.push('{');
    s.push_str("\"defaults\":{");
    s.push_str("\"enabled\":true,");
    s.push_str("\"roles\":[\"reader\",\"writer\"]");
    s.push_str("},\"items\":[");

    let mut index = 0usize;
    let mut first = true;
    while s.len() < target_size {
        if !first { s.push(','); } else { first = false; }
        // Build notes array for this item first
        let mut notes = String::with_capacity(notes_per_item * 64);
        notes.push('[');
        for note_index in 0..notes_per_item {
            if note_index != 0 { notes.push(','); }
            notes.push_str(&format!(
                "\"Note {:02} for item {:05}. This is repeated content to enlarge the JSON payload size considerably.\"",
                note_index, index
            ));
        }
        notes.push(']');

        // Build one item as JSON
        s.push('{');
        s.push_str(&format!("\"enabled\":true,\"roles\":[\"reader\",\"writer\"],\"id\":{},", index));
        s.push_str(&format!("\"name\":\"item_{:05}\",", index));
        s.push_str("\"details\":{");
        s.push_str(&format!(
            "\"description\":\"Item number {:05} includes repeated notes for benchmarking performance.\",",
            index
        ));
        s.push_str("\"notes\":");
        s.push_str(&notes);
        s.push('}'); // end details
        s.push('}'); // end item
        index += 1;
    }

    s.push(']');
    s.push('}');
    s
}

// --------------------- Parser adapters ---------------------

#[allow(dead_code)]
fn parse_serde_json(json: &str) -> Document {
    let doc: Document = serde_json::from_str(black_box(json)).expect("serde_json parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yaml(json: &str) -> Document {
    use serde_yaml as SY;
    let doc: Document = SY::from_str(black_box(json)).expect("serde_yaml parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yaml_ng(json: &str) -> Document {
    use serde_yaml_ng as SY;
    let doc: Document = SY::from_str(black_box(json)).expect("serde_yaml_ng parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yaml_bw(json: &str) -> Document {
    use serde_yaml_bw as SYBW;
    let doc: Document = SYBW::from_str(black_box(json)).expect("serde_yaml_bw parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yaml_norway(json: &str) -> Document {
    use serde_norway as SYN;
    let doc: Document = SYN::from_str(black_box(json)).expect("serde_yaml_norway parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yml(json: &str) -> Document {
    use serde_yml as SYML;
    let doc: Document = SYML::from_str(black_box(json)).expect("serde_yml parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_saphyr_budget_none(json: &str) -> Document {
    use serde_saphyr::{Error, Options};
    let opts = Options { budget: None, ..Options::default() };
    let doc: Result<Document, Error> = serde_saphyr::from_str_with_options(black_box(json), opts);
    let doc = doc.expect("serde_saphyr (budget=None) parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_saphyr_budget_max(json: &str) -> Document {
    use serde_saphyr::{budget::Budget, Error, Options};
    let many: usize = usize::MAX / 4;
    let opts = Options {
        budget: Some(Budget {
            max_reader_input_bytes: Some(many),
            max_events: many,
            max_aliases: many,
            max_anchors: many,
            max_depth: many,
            max_documents: many,
            max_nodes: many,
            max_total_scalar_bytes: many,
            max_merge_keys: many,
            enforce_alias_anchor_ratio: false,
            alias_anchor_min_aliases: many,
            alias_anchor_ratio_multiplier: many,
        }),
        ..Options::default()
    };
    let doc: Result<Document, Error> = serde_saphyr::from_str_with_options(black_box(json), opts);
    let doc = doc.expect("serde_saphyr (budget=max) parse failed");
    black_box(doc)
}

// --------------------- Criterion bench ---------------------

fn bench_compare_json(c: &mut Criterion) {
    // Sizes to sweep (MiB)
    let sizes_mib = [1usize, 5, 10, 25, 50];
    let notes_per_item = 20;

    let mut group = c.benchmark_group("json_parse");
    group.sampling_mode(SamplingMode::Auto);
    group.warm_up_time(Duration::from_secs(30));
    group.measurement_time(Duration::from_secs(600));
    group.sample_size(64);

    for &mib in &sizes_mib {
        let target_bytes = mib * 1024 * 1024;
        let json = build_large_json(target_bytes, notes_per_item);
        let json_len = json.len() as u64;
        group.throughput(Throughput::Bytes(json_len));

        group.bench_with_input(
            BenchmarkId::new("serde_json", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_serde_json(j); black_box(doc.items.len()); }),
        );

        // YAML-family parsers parsing JSON (YAML 1.2 is a superset of JSON)
        group.bench_with_input(
            BenchmarkId::new("serde_yaml", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_serde_yaml(j); black_box(doc.items.len()); }),
        );

        group.bench_with_input(
            BenchmarkId::new("serde_yaml_ng", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_serde_yaml_ng(j); black_box(doc.items.len()); }),
        );

        group.bench_with_input(
            BenchmarkId::new("serde_yaml_bw", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_serde_yaml_bw(j); black_box(doc.items.len()); }),
        );

        group.bench_with_input(
            BenchmarkId::new("serde_yaml_norway", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_serde_yaml_norway(j); black_box(doc.items.len()); }),
        );

        group.bench_with_input(
            BenchmarkId::new("serde_yml", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_serde_yml(j); black_box(doc.items.len()); }),
        );

        group.bench_with_input(
            BenchmarkId::new("serde_saphyr/budget_none", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_saphyr_budget_none(j); black_box(doc.items.len()); }),
        );

        group.bench_with_input(
            BenchmarkId::new("serde_saphyr/budget_max", format!("{}MiB", mib)),
            &json,
            |b, j| b.iter(|| { let doc = parse_saphyr_budget_max(j); black_box(doc.items.len()); }),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compare_json);
criterion_main!(benches);
