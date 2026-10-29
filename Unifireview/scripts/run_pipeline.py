"""
End-to-end orchestration script:
1. Preprocess raw reviews (dedup, normalization, SQLite + stats)
2. Run LLM batch analysis (summary, sentiment, themes) with telemetry
3. Perform sentence-level grounding
4. Emit weekly sentiment drift and telemetry rollups
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Add parent directory to Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from src.analysis.drift import weekly_sentiment
from src.grounding.semantic_grounding import GroundingThresholds, SemanticGrounder
from src.llm_analysis.ollama_analyzer import OllamaAnalyzer
from src.preprocessing.pipeline import preprocess
from src.telemetry.gpu_logger import TelemetryRecorder

# Cloud analyzer support
try:
    from src.llm_analysis.cloud_analyzer import CloudAnalyzer
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

# Ollama Cloud analyzer support
try:
    from src.llm_analysis.ollama_cloud_analyzer import OllamaCloudAnalyzer
    OLLAMA_CLOUD_AVAILABLE = True
except ImportError:
    OLLAMA_CLOUD_AVAILABLE = False


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def chunk_records(records: List[Dict], size: int) -> List[List[Dict]]:
    return [records[i : i + size] for i in range(0, len(records), size)]


def run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    start_time = time.time()

    print("\n" + "="*80)
    print(" REVIEW ANALYSIS PIPELINE - Starting")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Output: {args.results_out}")
    print()

    print("📁 STEP 1: Preprocessing reviews...")
    cleaned = preprocess(args.config)
    if cleaned.empty:
        print("❌ No reviews available after preprocessing. Outputs already written.")
        return
    print(f"✅ Loaded {len(cleaned)} reviews from {cleaned['platform'].nunique()} platform(s)")
    print(f"   Platforms: {', '.join(cleaned['platform'].unique())}")
    print()

    batch_size = config.get("pipeline", {}).get("batch_size", 50)
    print(f"📊 Batch size: {batch_size} reviews per batch")
    total_batches = (len(cleaned) + batch_size - 1) // batch_size
    print(f"📊 Total batches to process: {total_batches}")
    print()

    # Auto-detect analyzer type: Ollama Cloud > Cloud APIs > Local Ollama
    ollama_cloud_enabled = config.get("llm", {}).get("ollama_cloud", {}).get("enabled", False)
    cloud_enabled = config.get("llm", {}).get("cloud", {}).get("enabled", False)

    print("🤖 STEP 2: Initializing LLM analyzer...")
    if ollama_cloud_enabled and OLLAMA_CLOUD_AVAILABLE:
        print("☁️  Using Ollama Cloud (gpt-oss:120b, deepseek:671b, etc.)")
        analyzer = OllamaCloudAnalyzer(args.config)
    elif cloud_enabled and CLOUD_AVAILABLE:
        print("🌥️  Using cloud analyzer (Google Gemini/Groq/OpenAI)")
        analyzer = CloudAnalyzer(args.config)
    else:
        if ollama_cloud_enabled and not OLLAMA_CLOUD_AVAILABLE:
            print("⚠️  Ollama Cloud requested but not available, falling back to local")
        if cloud_enabled and not CLOUD_AVAILABLE:
            print("⚠️  Cloud requested but not available, falling back to local Ollama")
        print("🖥️  Using local Ollama analyzer")
        analyzer = OllamaAnalyzer(args.config, use_alt=args.alt_model)
    print(f"✅ Models configured:")
    for task, model in analyzer.models.items():
        print(f"   - {task}: {model}")
    print()
    grounding_cfg = config.get("grounding", {})
    thresholds = GroundingThresholds(
    flag=grounding_cfg.get("thresholds", {}).get("flag", 0.5),
    strong=grounding_cfg.get("thresholds", {}).get("strong", 0.7),
    )
    grounder = SemanticGrounder(
        model_name=grounding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        thresholds=thresholds,
    )
    print("🔧 STEP 3: Initializing grounding and telemetry...")
    telemetry = TelemetryRecorder(sample_interval=0.5)
    print(f"✅ Grounding model: {grounding_cfg.get('model', 'sentence-transformers/all-MiniLM-L6-v2')}")
    print(f"✅ Telemetry recording enabled")
    print()

    sentiment_records: List[Dict] = []
    batches: List[Dict] = []
    batch_counter = 0
    telemetry_summary = {"elapsed_hours": 0.0, "avg_power_w": None, "electricity_cost": None, "batches": []}

    print("⚙️  STEP 4: Processing batches...")
    print("="*80)

    try:
        for platform, group in cleaned.groupby("platform"):
            platform_records = group.to_dict("records")
            platform_batches = list(chunk_records(platform_records, batch_size))
            print(f"\n📱 Platform: {platform} ({len(platform_records)} reviews, {len(platform_batches)} batches)")

            for local_idx, batch_records in enumerate(platform_batches):
                batch_start_time = time.time()
                print(f"\n  ⏳ Batch {batch_counter + 1}/{total_batches} (Platform: {platform}, Local: {local_idx + 1}/{len(platform_batches)})")
                print(f"     Processing {len(batch_records)} reviews...")

                telemetry.start_batch()

                # LLM Analysis
                print(f"     🤖 Running LLM analysis (sentiment, summary, themes)...")
                batch_output = analyzer.analyze_batch(batch_records, batch_counter)
                print(f"     ✅ LLM complete ({batch_output['elapsed_s']:.1f}s)")

                telemetry_data = telemetry.end_batch(
                    label=f"{platform}-batch-{local_idx}",
                    elapsed_s=batch_output["elapsed_s"],
                )

                review_lookup = {row["review_id"]: row for row in batch_records}
                for sentiment in batch_output["sentiments"]:
                    review = review_lookup.get(sentiment["review_id"])
                    if review:
                        sentiment.update(
                            {
                                "rating_norm": review["rating_norm"],
                                "platform": review["platform"],
                                "review_date": review["review_date"],
                            }
                        )
                        sentiment_records.append(
                            {
                                "review_id": sentiment["review_id"],
                                "platform": review["platform"],
                                "review_date": review["review_date"],
                                "sentiment": sentiment["sentiment"],
                            }
                        )

                # Grounding
                print(f"     🎯 Running semantic grounding...")
                grounding = grounder.assess(
                    batch_output["summary"],
                    [row["review_text"] for row in batch_records],
                )
                print(f"     ✅ Grounding complete (score: {grounding.get('score', 0):.3f})")

                batch_output["grounding"] = grounding
                batch_output["source_reviews"] = [
                    {
                        "review_id": row["review_id"],
                        "review_text": row["review_text"],
                        "rating_norm": row["rating_norm"],
                    }
                    for row in batch_records
                ]
                batch_output["telemetry"] = telemetry_data.__dict__
                batch_output["platform"] = platform
                batch_output["platform_batch_index"] = local_idx
                batches.append(batch_output)
                batch_counter += 1

                batch_total_time = time.time() - batch_start_time
                elapsed_so_far = (time.time() - start_time) / 60
                avg_time_per_batch = elapsed_so_far / batch_counter
                remaining_batches = total_batches - batch_counter
                eta_minutes = avg_time_per_batch * remaining_batches

                print(f"     ⏱️  Batch time: {batch_total_time:.1f}s | Elapsed: {elapsed_so_far:.1f}m | ETA: {eta_minutes:.1f}m")

        print("\n" + "="*80)
        print(f"✅ All batches processed! ({batch_counter} batches, {len(sentiment_records)} reviews)")
        print("="*80)
    finally:
        print("\n📊 STEP 5: Finalizing telemetry and analysis...")
        elapsed_hours = (time.time() - start_time) / 3600.0
        telemetry_summary = telemetry.summarize(
            price_per_kwh=config.get("evaluation", {}).get("electricity_cost_per_kwh", 0.12),
            output_path=Path("data/results/telemetry.json"),
            elapsed_hours=elapsed_hours,
        )
        telemetry.close()
        print(f"✅ Telemetry saved: data/results/telemetry.json")

    print("\n📈 STEP 6: Calculating weekly sentiment drift...")
    weekly_sentiment(
        sentiment_records,
        Path("data/results/weekly_sentiment.csv"),
    )
    print(f"✅ Weekly sentiment saved: data/results/weekly_sentiment.csv")

    print("\n💾 STEP 7: Saving final results...")
    output_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "models_requested": analyzer.models,
        "models_used": analyzer.latest_models,
        "parameters": analyzer.generation,
        "batch_size": batch_size,
        "batches": batches,
        "telemetry_summary": telemetry_summary,
    }
    output_path = Path(args.results_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)

    print(f"✅ Results saved: {args.results_out}")
    print()
    print("="*80)
    print(" PIPELINE COMPLETE!")
    print("="*80)
    print(f"📊 Total reviews processed: {len(sentiment_records)}")
    print(f"📊 Total batches: {batch_counter}")
    print(f"⏱️  Total time: {elapsed_hours:.2f} hours ({elapsed_hours*60:.1f} minutes)")
    print(f"📁 Output file: {args.results_out}")
    print("="*80)
    print()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run review analysis pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--alt-model", action="store_true", help="Force alternate model use.")
    parser.add_argument(
        "--results-out",
        default="data/results/analysis_results.json",
        help="Path for analysis JSON output.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
