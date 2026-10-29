import csv
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PLATFORM_CONFIGS = {
    "trustpilot": {"category": "electronics", "product": "iPhone 15 Pro"},
    "yelp": {"category": "hospitality", "product": "The French Laundry"},
    "google": {"category": "services", "product": "Pixel Repair Center"},
}

TEXT_SNIPPETS = [
    "Battery life exceeds expectations and camera shines.",
    "Service team resolved my issue within a day.",
    "Menu creativity left us speechless.",
    "Room was spotless and staff attentive.",
    "Checkout process was confusing but manageable.",
    "Packaging arrived damaged though product worked fine.",
    "Live music elevated the entire evening.",
    "Parking was limited near the entrance.",
    "Portions were generous and plating artistic.",
    "Manager followed up personally to ensure satisfaction.",
]

FEATURES = [
    "support responsiveness",
    "ambient lighting",
    "chef tasting menu",
    "pro-level video capture",
    "in-store pickup timing",
    "table turnover cadence",
    "line management system",
    "CRM follow-up workflow",
    "sustainability packaging",
    "accessibility options",
]
REVIEWER_BASE = [
    "Alice",
    "Bob",
    "Carlos",
    "Dana",
    "Ella",
    "Frank",
    "Grace",
    "Hector",
    "Ivy",
    "Jon",
    "Kara",
    "Leo",
    "Mika",
    "Noah",
    "Omar",
    "Pia",
    "Quinn",
    "Ravi",
    "Sara",
    "Trent",
    "Uma",
    "Vik",
    "Wade",
    "Xena",
    "Yara",
    "Zane",
]

ADJECTIVES = [
    "luminous",
    "minimalist",
    "immersive",
    "frustrating",
    "seamless",
    "aspirational",
    "nostalgic",
    "playful",
    "nostalgic",
    "refined",
]

FRAGMENTS = [
    "Guests praised the lighting design that shifted through the night.",
    "Engineers cited the modem stability during roaming scenarios.",
    "Bar staff orchestrated flawless wine pairings without prompting.",
    "Logistics teams kept back orders under twelve hours despite spikes.",
    "Executive outreach included proactive make-goods and courtesy calls.",
    "Push notifications were throttled to respect quiet hours.",
    "Outdoor queue monitors dimmed per local ordinance yet remained legible.",
    "Firmware patched quietly and required no user action.",
    "Brand ambassadors captured testimonials for upcoming ad spots.",
    "On-site signage referenced QR workflows with minimal friction.",
    "AI-driven seating plans balanced VIP tables and walk-ins.",
    "Servers narrated sourcing stories for seasonal dishes.",
    "Repair bays swapped modules in under thirty minutes.",
    "App onboarding surfaced local perks and loyalty boosts.",
    "Kitchen expo pass tracked plating dwell times precisely.",
    "Warranty language was rewritten with plain English examples.",
    "Acoustic ceiling panels softened open-kitchen noise.",
    "Tablets at each booth offered co-browsing with staff.",
    "Sommeliers logged tasting notes for internal guild forums.",
    "CSR dashboards highlighted churn-risk customers nightly.",
    "Atrium scent design referenced regional botanicals.",
    "Chef's counter livestream fed TikTok content calendar.",
    "Back-of-house robots ferried linens to mezzanine storage.",
    "Charging docks lit up to match Apple aesthetic cues.",
    "Pre-shift briefings used VR roleplay scenarios.",
    "App waitlist gave ETA updates accurate to five minutes.",
    "Beverage flights mirrored sustainability commitments.",
    "In-store theater replayed keynote snippets on loop.",
    "Tableware showcased local ceramicist collaborations.",
    "Webhook integration piped sentiment to Slack war room.",
    "Roof garden herbs infused mocktails and amuse-bouche.",
    "Zero-touch checkout balanced fraud prevention with UX.",
    "Feedback kiosks offered multi-language voice input.",
    "Barista latte art celebrated device launch slogans.",
    "Accessibility docents coordinated with city advocates.",
    "Heads-up displays guided technicians through repairs.",
    "Dessert cart highlight was nitrogen-infused orchard fruit.",
    "CRM segmentation targeted cross-platform superfans.",
    "Guest ledger tracked allergies and celebration notes.",
    "Wayfinding projected onto floors during peak hours.",
]

VOCAB = [
    "atlas",
    "binary",
    "cobalt",
    "dynamo",
    "ember",
    "flux",
    "gadget",
    "harbor",
    "ionic",
    "junction",
    "keystone",
    "lumen",
    "matrix",
    "nebula",
    "oblique",
    "photon",
    "quartz",
    "ripple",
    "solstice",
    "trilogy",
    "ultra",
    "vector",
    "wharf",
    "xenon",
    "yonder",
    "zenith",
    "alpine",
    "beacon",
    "circuit",
    "delta",
    "ember",
    "fabric",
    "groove",
    "helium",
    "insight",
    "jigsaw",
    "krypton",
    "ledger",
    "monsoon",
    "nylon",
    "onyx",
    "pulsar",
    "quiver",
    "relay",
    "spectrum",
    "topaz",
    "umbra",
    "verge",
    "waveline",
    "xylem",
    "yacht",
    "zigzag",
    "archway",
    "blueprint",
    "cipher",
    "drift",
    "equinox",
    "feather",
    "granite",
    "harvest",
    "inkwell",
    "journal",
    "kiln",
    "lantern",
    "murmur",
    "nomad",
    "orbit",
    "paragon",
    "quartzite",
    "reservoir",
    "saber",
    "turquoise",
    "uplink",
    "vintage",
    "workshop",
    "yodel",
    "zephyr",
]


def build_rows(platform: str, count: int = 60):
    rows = []
    config = PLATFORM_CONFIGS[platform]
    for idx in range(count):
        day = (idx % 20) + 10
        review_date = f"2024-10-{day:02d}"
        rating = random.choice([1, 2, 3, 4, 5])
        reviewer = f"{REVIEWER_BASE[idx % len(REVIEWER_BASE)]} {platform[:2].upper()}{idx}"
        snippet = random.choice(TEXT_SNIPPETS)
        feature_set = random.sample(FEATURES, k=3)
        adjectives = random.sample(ADJECTIVES, k=2)
        unique_token = f"{platform[:2]}-{idx:03d}-{random.randint(1000,9999)}"
        noise = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=20))
        clause_pool = random.sample(FRAGMENTS, k=5)
        clauses = [
            snippet,
            f"The {config['product']} felt {adjectives[0]} yet {adjectives[1]} according to the guest.",
            f"They dwelled on {feature_set[0]}, noted {feature_set[1]}, and contrasted {feature_set[2]} with prior visits.",
            f"Unique observation token {unique_token} anchored the narrative for auditors.",
            f"Noise marker {noise} ensured lexical distance for benchmarking.",
            *clause_pool,
        ]
        random.shuffle(clauses)
        word_stream = " ".join(random.choices(VOCAB, k=40))
        text = " ".join(clauses + [word_stream])
        rows.append(
            {
                "platform": platform,
                "category": config["category"],
                "product_name": config["product"],
                "rating": rating,
                "review_text": text,
                "review_date": review_date,
                "reviewer_name": reviewer,
                "verified_purchase": random.choice(["true", "false"]),
            }
        )
    return rows


def main():
    for platform in PLATFORM_CONFIGS:
        rows = build_rows(platform)
        path = RAW_DIR / f"{platform}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
