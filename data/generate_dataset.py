"""
generate_dataset.py
--------------------
Builds an ORIGINAL, fully synthetic customer-support email/reply dataset for
ReplyLens AI.

Why synthetic, template-based generation (no scraping, no LLM calls needed)?
- It is reproducible: a fixed random seed always regenerates the same file.
- It avoids any privacy/IP concerns tied to real customer or company emails.
- It gives us direct control over category balance, which matters for a
  meaningful evaluation benchmark later.

The generator works by combining:
  1) A pool of fictional customer names, order numbers, product names, dates.
  2) Category-specific email/reply "templates" with placeholders.
  3) Light randomized variation in phrasing, so records in the same category
     are not identical copies of each other.

Output: data/emails.csv with columns:
  id, category, incoming_email, historical_reply, intent, resolution_type, split

`split` is either "reference" (80%) or "evaluation" (20%). The evaluation
split is used ONLY for benchmark scoring; retrieval never pulls examples
from it, which avoids retrieval leakage into the evaluation numbers.
"""

import csv
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

OUT_PATH = Path(__file__).parent / "emails.csv"

FIRST_NAMES = [
    "Alex", "Priya", "Jordan", "Maria", "Sam", "Wei", "Fatima", "Lucas",
    "Emma", "Noah", "Aisha", "Diego", "Chloe", "Ravi", "Sofia", "Liam",
    "Grace", "Omar", "Hana", "Ethan",
]

PRODUCTS = [
    "wireless headphones", "standing desk", "coffee grinder", "running shoes",
    "yoga mat", "office chair", "blender", "backpack", "desk lamp",
    "bluetooth speaker", "laptop stand", "water bottle", "air purifier",
    "smart watch", "notebook set",
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def rand_name():
    return random.choice(FIRST_NAMES)


def rand_order_id():
    return f"ORD-{random.randint(10000, 99999)}"


def rand_product():
    return random.choice(PRODUCTS)


def rand_date():
    return f"{random.choice(MONTHS)} {random.randint(1, 28)}"


def rand_amount():
    return f"${random.choice([19, 29, 39, 49, 59, 79, 99, 129, 149, 199])}.{random.choice(['00', '99'])}"


# Each category maps to a list of (email_templates, reply_templates, intent, resolution_type)
# Templates use {name}, {order}, {product}, {date}, {amount} placeholders.
CATEGORIES = {
    "refund_request": {
        "intent": "request a refund for a purchase",
        "resolution_type": "refund_initiated",
        "emails": [
            "Hi, I'd like to request a refund for order {order}. The {product} I received doesn't work as expected. Can you help?",
            "Hello, I'm not happy with my recent purchase ({order}). I'd like a full refund for the {product}, please.",
            "Hi team, I purchased a {product} on {date} (order {order}) and would like to return it for a refund.",
        ],
        "replies": [
            "Hi {name}, thanks for reaching out. I'm sorry the {product} didn't meet your expectations. I've initiated a refund for order {order}; you should see the amount back in your account within 5-7 business days.",
            "Hello {name}, I understand the frustration. I've processed a refund request for order {order}. Please allow 5-7 business days for the funds to appear on your original payment method.",
        ],
    },
    "order_status": {
        "intent": "check the current status of an order",
        "resolution_type": "status_provided",
        "emails": [
            "Hi, can you tell me the status of my order {order}? I placed it on {date} and haven't heard anything.",
            "Hello, I'd like an update on order {order} for the {product}. Is it still on track?",
            "Hi there, just checking in on order {order} placed {date}. Any updates?",
        ],
        "replies": [
            "Hi {name}, thanks for checking in. Order {order} is currently being processed and is on track to ship soon. You'll receive a tracking number by email once it leaves our warehouse.",
            "Hello {name}, I checked on order {order} for you — it's in our fulfillment queue and should ship within 1-2 business days.",
        ],
    },
    "delayed_delivery": {
        "intent": "report that a delivery is delayed and ask for an update",
        "resolution_type": "escalation_or_update",
        "emails": [
            "Hi, my order {order} was supposed to arrive by {date} but it still hasn't shown up. Can you check what's going on?",
            "Hello, the {product} I ordered ({order}) is significantly delayed. Where is it?",
            "Hi team, order {order} is past its estimated delivery date. Can you look into this?",
        ],
        "replies": [
            "Hi {name}, I'm sorry for the delay with order {order}. I've checked with our shipping partner and it looks like there was a delay in transit. I've flagged this for priority handling and will follow up with an updated delivery date shortly.",
            "Hello {name}, apologies for the inconvenience. Order {order} is currently delayed in transit. I've escalated this to our logistics team and will update you as soon as I have a new estimated delivery date.",
        ],
    },
    "damaged_product": {
        "intent": "report that a received product arrived damaged",
        "resolution_type": "replacement_or_refund",
        "emails": [
            "Hi, I received my {product} today (order {order}) but it arrived damaged. What are my options?",
            "Hello, the {product} from order {order} came with visible damage during shipping. Can you help?",
            "Hi, unfortunately my order {order} arrived broken. I'd like a replacement or refund.",
        ],
        "replies": [
            "Hi {name}, I'm sorry to hear the {product} arrived damaged — that's not the experience we want for you. I can send a free replacement or process a full refund for order {order}, whichever you'd prefer.",
            "Hello {name}, apologies for the damage on order {order}. I've started a replacement shipment for you at no extra cost, and you won't need to return the damaged item.",
        ],
    },
    "cancellation": {
        "intent": "cancel an order before it ships",
        "resolution_type": "order_cancelled",
        "emails": [
            "Hi, I need to cancel order {order} before it ships. Is that still possible?",
            "Hello, please cancel my order for the {product} ({order}). I no longer need it.",
            "Hi team, can you cancel order {order}? I placed it by mistake.",
        ],
        "replies": [
            "Hi {name}, I've cancelled order {order} for you. Since it hadn't shipped yet, no charges will be applied and any pending authorization will be released within a few business days.",
            "Hello {name}, order {order} has been successfully cancelled. You won't be charged, and you'll receive a confirmation email shortly.",
        ],
    },
    "account_access": {
        "intent": "resolve an issue accessing their account",
        "resolution_type": "access_restored_or_steps_provided",
        "emails": [
            "Hi, I can't log into my account anymore. It keeps saying my credentials are invalid. Can you help?",
            "Hello, I'm locked out of my account after a few failed login attempts. What should I do?",
            "Hi team, my account seems to have been suspended and I don't know why.",
        ],
        "replies": [
            "Hi {name}, sorry for the trouble logging in. I've unlocked your account — please try logging in again, and use the 'Forgot Password' link if you'd like to set a new password.",
            "Hello {name}, I've reviewed your account and lifted the temporary lock caused by repeated login attempts. You should be able to sign in normally now.",
        ],
    },
    "password_reset": {
        "intent": "request help resetting their password",
        "resolution_type": "steps_provided",
        "emails": [
            "Hi, I forgot my password and the reset email isn't arriving. Can you help me regain access?",
            "Hello, I'd like to reset my password but I'm not receiving the reset link. What can I do?",
            "Hi team, how do I reset my account password? I tried the usual link with no luck.",
        ],
        "replies": [
            "Hi {name}, I've manually triggered a password reset email to the address on file — please check your inbox and spam folder over the next few minutes. Let me know if it still doesn't arrive.",
            "Hello {name}, sorry about that. I've resent the password reset link to your registered email. If it doesn't show up shortly, I can reset it manually from our end.",
        ],
    },
    "billing_issue": {
        "intent": "resolve a billing discrepancy",
        "resolution_type": "billing_corrected_or_explained",
        "emails": [
            "Hi, I noticed a charge on my statement I don't recognize for order {order}. Can you explain this?",
            "Hello, my invoice for {product} shows {amount} but I was expecting a different amount. Can you clarify?",
            "Hi team, there seems to be a billing error on my account related to order {order}.",
        ],
        "replies": [
            "Hi {name}, thanks for flagging this. I looked into order {order} and the {amount} charge reflects the item price plus applicable tax. I've attached a breakdown for clarity — let me know if anything still looks off.",
            "Hello {name}, I reviewed your billing history and found the discrepancy. I've corrected it on our end and you should see an updated statement within one billing cycle.",
        ],
    },
    "duplicate_charge": {
        "intent": "report being charged twice for the same order",
        "resolution_type": "duplicate_refunded",
        "emails": [
            "Hi, I was charged twice for order {order}. Can you refund the duplicate charge?",
            "Hello, I see two identical charges of {amount} on my card for the same {product} order. Please help.",
            "Hi team, order {order} appears to have been billed twice by mistake.",
        ],
        "replies": [
            "Hi {name}, I'm sorry for the duplicate charge on order {order}. I've confirmed it was billed twice in error and have refunded the extra charge; it should appear in 5-7 business days.",
            "Hello {name}, thanks for letting us know. I've verified the duplicate charge and processed a refund for the second charge on order {order}.",
        ],
    },
    "subscription_cancellation": {
        "intent": "cancel a recurring subscription",
        "resolution_type": "subscription_cancelled",
        "emails": [
            "Hi, I'd like to cancel my subscription. Please make sure I'm not billed again.",
            "Hello, please cancel my recurring plan effective immediately.",
            "Hi team, how do I cancel my subscription before the next billing date?",
        ],
        "replies": [
            "Hi {name}, I've cancelled your subscription effective immediately. You won't be billed going forward, and you'll retain access until the end of your current billing period.",
            "Hello {name}, your subscription has been cancelled as requested. No further charges will be made, and you'll receive a confirmation email shortly.",
        ],
    },
    "product_information": {
        "intent": "ask for more information about a product before purchasing",
        "resolution_type": "information_provided",
        "emails": [
            "Hi, can you tell me more about the {product}? Specifically, I'd like to know about sizing and materials.",
            "Hello, does the {product} come with a warranty? I'm considering buying one.",
            "Hi team, I have a few questions about the {product} before I order it.",
        ],
        "replies": [
            "Hi {name}, happy to help! The {product} comes with a 1-year limited warranty and detailed specs are listed on the product page. Let me know if you have any other questions before ordering.",
            "Hello {name}, thanks for your interest in the {product}. It's available in multiple sizes and includes care instructions in the box. I'm happy to answer any other questions.",
        ],
    },
    "complaint": {
        "intent": "voice a general complaint about service or experience",
        "resolution_type": "acknowledged_and_escalated",
        "emails": [
            "Hi, I'm really unhappy with the service I received regarding order {order}. This isn't the experience I expected.",
            "Hello, I want to formally complain about how my recent order ({order}) was handled.",
            "Hi team, I've had a frustrating experience with support regarding {product} and want this addressed.",
        ],
        "replies": [
            "Hi {name}, I'm sorry to hear about your experience with order {order} — that falls short of what we aim for. I've escalated this to our support lead and will personally follow up to make sure it's resolved.",
            "Hello {name}, thank you for your patience and for letting us know. I've logged your feedback about order {order} and escalated it internally so we can prevent this going forward.",
        ],
    },
    "technical_support": {
        "intent": "get help with a technical problem using a product or service",
        "resolution_type": "troubleshooting_provided",
        "emails": [
            "Hi, my {product} isn't connecting properly. I've tried restarting it but no luck. Can you help?",
            "Hello, I'm having trouble setting up my {product}. The instructions don't seem to match what I'm seeing.",
            "Hi team, the app keeps crashing when I try to pair it with my {product}.",
        ],
        "replies": [
            "Hi {name}, sorry for the trouble. Could you try resetting the {product} by holding the power button for 10 seconds, then reconnecting via Bluetooth? Let me know if that resolves it or if you'd like a video walkthrough.",
            "Hello {name}, thanks for the details. This is usually caused by outdated firmware — could you update the {product}'s firmware through the companion app and try again?",
        ],
    },
    "address_change": {
        "intent": "update or correct a shipping address",
        "resolution_type": "address_updated",
        "emails": [
            "Hi, I need to update the shipping address for order {order} before it ships. Can you help?",
            "Hello, I entered the wrong address on order {order}. Can this be corrected?",
            "Hi team, please change the delivery address on my order {order}.",
        ],
        "replies": [
            "Hi {name}, I've updated the shipping address for order {order}. Since it hasn't shipped yet, the new address will be used for delivery.",
            "Hello {name}, good news — order {order} hadn't shipped yet, so I was able to update the address as requested.",
        ],
    },
    "exchange_request": {
        "intent": "exchange a purchased item for a different size or variant",
        "resolution_type": "exchange_initiated",
        "emails": [
            "Hi, I'd like to exchange the {product} from order {order} for a different size. How do I do that?",
            "Hello, the {product} I received doesn't fit as expected. Can I exchange it for another size?",
            "Hi team, can I swap the {product} in order {order} for a different color?",
        ],
        "replies": [
            "Hi {name}, of course! I've started an exchange for order {order} — a prepaid return label is on its way to your email, and we'll ship the replacement as soon as the original is on its way back to us.",
            "Hello {name}, happy to help with the exchange. I've arranged for a new {product} to be shipped and included a prepaid label for returning the original from order {order}.",
        ],
    },
}


def build_records():
    records = []
    rid = 1
    # Aim for ~26-27 records per category => ~400 total across 15 categories
    per_category = 27
    for category, spec in CATEGORIES.items():
        for _ in range(per_category):
            email_t = random.choice(spec["emails"])
            reply_t = random.choice(spec["replies"])
            ctx = {
                "name": rand_name(),
                "order": rand_order_id(),
                "product": rand_product(),
                "date": rand_date(),
                "amount": rand_amount(),
            }
            incoming = email_t.format(**ctx)
            reply = reply_t.format(**ctx)
            records.append({
                "id": f"E{rid:04d}",
                "category": category,
                "incoming_email": incoming,
                "historical_reply": reply,
                "intent": spec["intent"],
                "resolution_type": spec["resolution_type"],
            })
            rid += 1
    return records


def assign_splits(records, eval_fraction=0.2):
    # Deterministic split: shuffle with fixed seed, then take a contiguous
    # slice as "evaluation". Stratified per category to keep balance.
    by_category = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)

    for cat_records in by_category.values():
        random.shuffle(cat_records)
        n_eval = max(1, int(len(cat_records) * eval_fraction))
        for i, r in enumerate(cat_records):
            r["split"] = "evaluation" if i < n_eval else "reference"
    return records


def main():
    records = build_records()
    records = assign_splits(records)
    random.shuffle(records)

    fieldnames = ["id", "category", "incoming_email", "historical_reply",
                  "intent", "resolution_type", "split"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    n_ref = sum(1 for r in records if r["split"] == "reference")
    n_eval = sum(1 for r in records if r["split"] == "evaluation")
    print(f"Wrote {len(records)} records to {OUT_PATH}")
    print(f"  reference: {n_ref}, evaluation: {n_eval}")
    print(f"  categories: {len(CATEGORIES)}")


if __name__ == "__main__":
    main()
