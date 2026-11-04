# Competition Clarifications

> [!IMPORTANT]
> The rules below **take precedence** over any other information you may have received previously, whether in the documentation, in earlier emails, or in answers to individual questions.

---

## What is not allowed

- **Checking your output against the video.**
- **Hand labels, human-produced data, or any other data created manually** on the provided game.
- **Uploading the provided data to external/commercial AI services** (e.g., cloud LLMs/APIs where data privacy/terms cannot be guaranteed).
- **Training, fitting, or calibrating your algorithm** on the ground-truth possession data or on the provided video.

---

## What is allowed

- **Running automatic checks** using only the provided data.
- **Inspecting the provided data and the video** to better understand what counts as possession.
- **Using as input the video, the automatic tracking data, the smart tagging, or any other generated data**, as long as your algorithm was not trained on the ground truth or on the provided video.
- **Providing files of intermediate steps.**
- **Using any dataset you have the rights to use.** *(Please verify the license of any dataset before using it).*

---

## Training and Tuning

> [!WARNING]
> - If you need to **train (or retrain)**, you **MUST** use only external data.
> - If your algorithm has parameters that need to be tuned, you **MUST** tune them on external data, **never** on the ground-truth possession data or on the provided video.
