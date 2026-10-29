"""
End-to-end pipeline with comprehensive audit logging.
Every step is logged to terminal with visual indicators.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.analysis.drift import weekly_sentiment
from src.grounding.semantic_grounding import GroundingThresholds, SemanticGrounder
from src.llm_analysis.ollama_analyzer import OllamaAnalyzer
from src.preprocessing.pipeline import preprocess
from src.telemetry.gpu_logger import TelemetryRecorder

from src.analysis.drift import weekly_sentiment
from src.grounding.semantic_grounding import GroundingThresholds, SemanticGrounder
from src.llm_analysis.ollama_analyzer import OllamaAnalyzer
from src.preprocessing.pipeline import preprocess
from src.telemetry.gpu_logger import TelemetryRecorder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/logs/pipeline_audit.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def print_section(title: str, char: str = "="):
    """Print a section header."""
    width = 80
    logger.info("")
    logger.info(char * width)
    logger.info(f" {title}")
    logger.info(char * width)


def print_step(step_num: int, total_steps: int, description: str):
    """Print a step indicator."""
    logger.info(f"\n[STEP {step_num}/{total_steps}] {description}")
    logger.info("-" * 80)


def print_success(message: str):
    """Print success message."""
    logger.info(f"✅ {message}")


def print_warning(message: str):
    """Print warning message."""
    logger.warning(f"⚠️  {message}")


def print_error(message: str):
    """Print error message."""
    logger.error(f"❌ {message}")


def print_info(message: str):
    """Print info message."""
    logger.info(f"ℹ️  {message}")


def load_config(path: Path) -> Dict:
    """Load configuration with audit log."""
    logger.info(f"📄 Loading configuration from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        print_success(f"Configuration loaded successfully")

        # Log key config parameters
        logger.info("📋 Configuration Summary:")
        logger.info(f"   - LLM Server: {config['llm']['server']['base_url']}")
        logger.info(f"   - Timeout: {config['llm']['server']['timeout']}s")
        logger.info(f"   - Batch Size: {config['pipeline']['batch_size']}")
        logger.info(f"   - Models: {config['llm']['models']}")

        return config
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        raise


def chunk_records(records: List[Dict], size: int) -> List[List[Dict]]:
    """Chunk records with audit log."""
    chunks = [records[i : i + size] for i in range(0, len(records), size)]
    logger.info(f"📦 Split {len(records)} records into {len(chunks)} batches of size {size}")
    return chunks


def run(args: argparse.Namespace) -> None:
    """Run pipeline with comprehensive audit logging."""
    print_section("MULTI-PLATFORM REVIEW ANALYSIS PIPELINE", "=")
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Config: {args.config}")

    total_steps = 9
    start_time = time.time()

    # STEP 1: Load Configuration
    print_step(1, total_steps, "Loading Configuration")
    config = load_config(Path(args.config))

    # STEP 2: Preprocessing
    print_step(2, total_steps, "Preprocessing Reviews")
    logger.info("🔧 Starting preprocessing pipeline...")
    logger.info("   - Loading raw CSVs")
    logger.info("   - Deduplicating reviews")
    logger.info("   - Normalizing ratings")
    logger.info("   - Filtering spam and non-English")

    try:
        cleaned = preprocess(args.config)
        print_success(f"Preprocessing complete: {len(cleaned)} reviews ready for analysis")
        logger.info(f"📊 Review Statistics:")
        logger.info(f"   - Total reviews: {len(cleaned)}")
        logger.info(f"   - Platforms: {cleaned['platform'].nunique()}")
        logger.info(f"   - Date range: {cleaned['review_date'].min()} to {cleaned['review_date'].max()}")
    except Exception as e:
        print_error(f"Preprocessing failed: {e}")
        raise

    if cleaned.empty:
        print_warning("No reviews available after preprocessing")
        return

    # STEP 3: Initialize Components
    print_step(3, total_steps, "Initializing Analysis Components")

    batch_size = config.get("pipeline", {}).get("batch_size", 50)
    logger.info(f"📦 Batch size: {batch_size}")

    logger.info("🤖 Initializing LLM analyzer...")
    try:
        analyzer = OllamaAnalyzer(args.config, use_alt=args.alt_model)
        print_success("LLM analyzer initialized")
        logger.info(f"   - Models: {analyzer.models}")
        logger.info(f"   - Base URL: {analyzer.base_url}")
        logger.info(f"   - Timeout: {analyzer.timeout}s")
    except Exception as e:
        print_error(f"Failed to initialize analyzer: {e}")
        raise

    logger.info("🔍 Initializing semantic grounder...")
    try:
        grounding_cfg = config.get("grounding", {})
        thresholds = GroundingThresholds(
            flag=grounding_cfg.get("thresholds", {}).get("flag", 0.5),
            strong=grounding_cfg.get("thresholds", {}).get("strong", 0.7),
        )
        grounder = SemanticGrounder(
            model_name=grounding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            thresholds=thresholds,
        )
        print_success(f"Semantic grounder initialized (device: {grounder.device})")
    except Exception as e:
        print_error(f"Failed to initialize grounder: {e}")
        raise

    logger.info("📊 Initializing telemetry recorder...")
    telemetry = TelemetryRecorder(sample_interval=0.5)
    print_success("Telemetry recorder initialized")

    # STEP 4: Batch Processing
    print_step(4, total_steps, "Processing Review Batches")

    sentiment_records: List[Dict] = []
    batches: List[Dict] = []
    batch_counter = 0
    telemetry_summary = {"elapsed_hours": 0.0, "avg_power_w": None, "electricity_cost": None, "batches": []}

    try:
        for platform_idx, (platform, group) in enumerate(cleaned.groupby("platform"), 1):
            logger.info(f"\n🏢 Processing platform {platform_idx}/{cleaned['platform'].nunique()}: {platform}")
            logger.info(f"   - Reviews: {len(group)}")

            platform_records = group.to_dict("records")

            for local_idx, batch_records in enumerate(chunk_records(platform_records, batch_size), 1):
                logger.info(f"\n   📦 Batch {local_idx} ({len(batch_records)} reviews)")
                logger.info(f"      Batch ID: {platform}-batch-{local_idx-1}")

                # Start telemetry
                telemetry.start_batch()
                logger.info("      ⏱️  Started batch analysis...")

                # Analyze batch
                try:
                    batch_output = analyzer.analyze_batch(batch_records, batch_counter)
                    print_success(f"      Batch {local_idx} analysis complete")

                    logger.info(f"      📝 Summary length: {len(batch_output['summary'])} chars")
                    logger.info(f"      🎨 Themes extracted: {len(batch_output['themes'])}")
                    logger.info(f"      😊 Sentiments analyzed: {len(batch_output['sentiments'])}")
                    logger.info(f"      ⏱️  Elapsed time: {batch_output['elapsed_s']:.2f}s")
                    logger.info(f"      🤖 Models used: {batch_output['models_used']}")

                except Exception as e:
                    print_error(f"      Batch {local_idx} failed: {e}")
                    raise

                # End telemetry
                telemetry_data = telemetry.end_batch(
                    label=f"{platform}-batch-{local_idx-1}",
                    elapsed_s=batch_output["elapsed_s"],
                )
                logger.info(f"      📊 Telemetry recorded")

                # Enrich sentiments with platform/date info
                review_lookup = {row["review_id"]: row for row in batch_records}
                for sentiment in batch_output["sentiments"]:
                    review = review_lookup.get(sentiment["review_id"])
                    if review:
                        sentiment.update({
                            "rating_norm": review["rating_norm"],
                            "platform": review["platform"],
                            "review_date": review["review_date"],
                        })
                        sentiment_records.append({
                            "review_id": sentiment["review_id"],
                            "platform": review["platform"],
                            "review_date": review["review_date"],
                            "sentiment": sentiment["sentiment"],
                        })

                # STEP 5: Semantic Grounding
                logger.info("      🔍 Performing semantic grounding...")
                try:
                    grounding = grounder.assess(
                        batch_output["summary"],
                        [row["review_text"] for row in batch_records],
                    )
                    logger.info(f"         - Strong: {grounding['strong']}")
                    logger.info(f"         - Medium: {grounding['medium']}")
                    logger.info(f"         - Weak: {grounding['weak']}")
                    print_success(f"      Grounding complete")
                except Exception as e:
                    print_warning(f"      Grounding failed: {e}")
                    grounding = {"strong": 0, "medium": 0, "weak": 0}

                # Compile batch output
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
                batch_output["platform_batch_index"] = local_idx - 1
                batches.append(batch_output)

                batch_counter += 1
                print_success(f"   ✅ Batch {local_idx} complete (total: {batch_counter} batches)")

    except KeyboardInterrupt:
        print_warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        raise
    finally:
        # STEP 6: Finalize Telemetry
        print_step(6, total_steps, "Finalizing Telemetry")
        elapsed_hours = (time.time() - start_time) / 3600.0
        logger.info(f"⏱️  Total elapsed time: {elapsed_hours:.4f} hours ({elapsed_hours * 60:.2f} minutes)")

        telemetry_summary = telemetry.summarize(
            price_per_kwh=config.get("evaluation", {}).get("electricity_cost_per_kwh", 0.12),
            output_path=Path("data/results/telemetry.json"),
            elapsed_hours=elapsed_hours,
        )
        telemetry.close()
        print_success("Telemetry finalized")

    # STEP 7: Weekly Sentiment Analysis
    print_step(7, total_steps, "Computing Weekly Sentiment Drift")
    logger.info(f"📊 Analyzing {len(sentiment_records)} sentiment records...")

    try:
        weekly_sentiment(
            sentiment_records,
            Path("data/results/weekly_sentiment.csv"),
        )
        print_success("Weekly sentiment analysis complete")
    except Exception as e:
        print_error(f"Weekly sentiment failed: {e}")
        raise

    # STEP 8: Generate Output
    print_step(8, total_steps, "Generating Output Files")

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

    logger.info(f"💾 Writing results to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_payload, fh, indent=2)
        print_success(f"Results saved to: {output_path}")
    except Exception as e:
        print_error(f"Failed to save results: {e}")
        raise

    # STEP 9: Summary
    print_step(9, total_steps, "Pipeline Summary")

    logger.info("📊 FINAL STATISTICS:")
    logger.info(f"   - Total reviews processed: {len(cleaned)}")
    logger.info(f"   - Total batches: {batch_counter}")
    logger.info(f"   - Platforms: {cleaned['platform'].nunique()}")
    logger.info(f"   - Sentiments analyzed: {len(sentiment_records)}")
    logger.info(f"   - Elapsed time: {elapsed_hours * 60:.2f} minutes")
    logger.info(f"   - Avg time per review: {(elapsed_hours * 3600 / len(cleaned)):.2f}s")

    logger.info("\n📁 OUTPUT FILES:")
    logger.info(f"   - Analysis results: {output_path}")
    logger.info(f"   - Cleaned reviews: data/processed/reviews_clean.csv")
    logger.info(f"   - Weekly sentiment: data/results/weekly_sentiment.csv")
    logger.info(f"   - Telemetry: data/results/telemetry.json")
    logger.info(f"   - Descriptive stats: data/results/descriptive.json")
    logger.info(f"   - Audit log: data/logs/pipeline_audit.log")

    print_section("PIPELINE COMPLETE", "=")
    print_success(f"Pipeline completed successfully in {elapsed_hours * 60:.2f} minutes!")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run review analysis pipeline with audit logging.")
    parser.add_argument("--config", default="config/config_alt.yaml", help="Path to config file")
    parser.add_argument("--alt-model", action="store_true", help="Force alternate model use")
    parser.add_argument(
        "--results-out",
        default="data/results/analysis_results.json",
        help="Path for analysis JSON output",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
