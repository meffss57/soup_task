# Dataset sources and method

The task package (~500 real Russian support-ticket preference pairs) was not provided, so this is a
**synthetic mock dataset** of Russian e-commerce customer-support tickets (orders, delivery,
refunds, payments, accounts, complaints). No real customer text is included.

## Open-source references

| Source | Licence | What it was used for |
|---|---|---|
| **Yandex Geo Reviews Dataset 2023**, [GitHub](https://github.com/yandex/geo-reviews-dataset-2023) / [HF mirror](https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023) | MIT | Real Russian complaint language. Filtered to rubrics *Пункт выдачи, Курьерские услуги, Офис интернет-магазина, Почтовые услуги* with rating ≤ 2 (1,051 reviews). Used to calibrate the noise rates (punctuation, `!!!`, CAPS, `...`/`((`, missing spaces) and to check the complaint topics. See `style_reference.py` and `soup_dpo_data/style_reference_report.json`. |
| **Support_Contact_Center_RU_27000** ([HF](https://huggingface.co/datasets/mkaiuse/Support_Contact_Center_RU_27000)), Russian translation of the Bitext customer-support dataset | Apache-2.0 | Cross-check that the categories cover standard support intents (cancel_order, track_order, get_refund, payment_issue, change_shipping_address, recover_password, delete_account, complaint, subscription…). Not used as text: its answers are machine-translated, templated and read as AI ("Я здесь, чтобы помочь вам!"), which is the style we avoid. |

## Category → support-intent mapping

| Our category | Standard intent |
|---|---|
| order_cancellation | cancel_order |
| delivery_late, delivery_tracking | delivery_period, track_order |
| delivery_address | change_shipping_address |
| refund_delay, refund_amount | track_refund, get_refund, check_refund_policy |
| return_request, damaged_item, wrong_item, missing_item, warranty | get_refund / complaint (product issues) |
| payment_failed, duplicate_charge, wrong_price, promo_code_issue | payment_issue, check_invoice |
| account_access, personal_data | recover_password, delete_account, edit_account |
| subscription | subscription |
| rude_staff, tech_app_crash | complaint, contact_customer_service |

## How pairs are built (`generate_soup_dpo_professional.py`)

- 20 categories × 25 = 500 pairs, fixed seed, stratified 400/100 split.
- **Prompt**: 13–17 hand-written ticket templates per category (angry one-liners to formal complaints),
  each used at most twice. Optional context sentence, greeting, tail and order number, then noise:
  typos on the ЙЦУКЕН layout, lowercase/autocapitalisation, lost punctuation, `!!!`/`((`, CAPS,
  chat-style line breaks, rare translit. Russian cities, banks and payment methods, amounts in ₽.
- **Chosen**: built from category fragments (acknowledge → concrete action → specific request → next step),
  filtered by ticket sub-case tags, with paraphrase variants. Uses the same item, amount and order as the ticket.
- **Rejected**: a *hard negative*. It is written with the same greeting, acknowledgement, ticket number,
  request sentence and operator sign-off as the chosen answer, on the same topic and in the same polite tone,
  but it is wrong in substance. Labelled in `metadata.failure_mode`:
  wrong_fact (wrong deadline / refund route / "impossible" when it is possible), passive_wait (no check, just wait),
  generic_instruction (self-service steps that ignore the specifics), polite_deflection (sends to seller / bank / carrier),
  unverified_promise ("already refunded" without checking), partial_resolution, wrong_subcase (a correct answer
  for a *different* sub-case of the same problem), redundant_question (asks for the order number the customer
  already gave), unsafe_data_request (politely asks for a full card number, SMS code or passport photo).
- **Checks**: no duplicate prompt/chosen/rejected, no near-duplicate prompts within a category (difflib ≥ 0.82),
  a blacklist of AI-sounding phrases, no unfilled slots, no train/eval prompt leakage.

## Known limitations (relevant to the take-home's "silent failures")

- **Shortcut baselines** (`dataset_report.json → shortcut_baselines`): "longer answer wins" is right on ~71% of pairs,
  and a TF-IDF + logistic-regression classifier (5-fold, grouped by pair) picks the chosen answer ~92% of the time.
  Part of this is legitimate (good answers commit to an action: "проверили", "вернем"; bad ones say "нельзя",
  "подождите", "обратитесь в банк"), part is template vocabulary. A model can therefore reach high preference
  accuracy without reading the ticket; that is exactly why a falling DPO loss is not proof of learning.
- **Template reuse**: 44/100 eval rows share a prompt template with a train row (different slots and noise).
- Synthetic tickets are shorter than public reviews (median ~90 vs ~370 chars). Real tickets would be messier and less balanced.
