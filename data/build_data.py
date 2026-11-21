"""Builds the synthetic dataset used in the article.

"Lumen Notes" is a fictional note-taking app. Every document and question here is
made up for the demo, so there is no real customer or personal data.

Outputs:
  data/docs.jsonl        - 24 help-center articles (the knowledge base)
  data/golden_set.jsonl  - 40 test questions with the doc(s) that answer them

Question types:
  keyword      - phrased with the same words the docs use
  paraphrase   - same intent, different vocabulary (how real users often ask)
  unanswerable - the knowledge base has no answer; a good system should say so
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

DOCS = [
    ("export-pdf", "Export notes to PDF",
     "You can export any note or notebook to PDF. Open the note, click the Share menu and choose Export as PDF. "
     "Images and tables are kept. Exporting a whole notebook creates one PDF with a table of contents. "
     "PDF export is available on all plans, including Free."),
    ("export-markdown", "Export notes to Markdown",
     "To export a note as Markdown, open Share and choose Export as Markdown. Attachments are saved in a folder next "
     "to the .md file. Bulk Markdown export of every notebook is available on Pro and Team plans from Settings > Data."),
    ("offline-mode", "Using Lumen Notes offline",
     "The desktop and mobile apps work offline. Notes you open are cached on your device, and edits made offline are "
     "synced automatically when you reconnect. To make a whole notebook available offline, right-click it and enable "
     "Keep offline. The web app requires an internet connection."),
    ("sync-conflicts", "Resolving sync conflicts",
     "A sync conflict happens when the same note is edited on two devices before they sync. Lumen Notes keeps both "
     "versions: the newest stays in place and the other is saved as a copy named 'Conflicted copy' with the device name "
     "and date. Compare the two and delete the one you don't need."),
    ("version-history", "Version history",
     "Lumen Notes saves a version of a note every time you stop editing for 30 seconds. Open the note, click the clock "
     "icon and pick a version to preview or restore it. Free plans keep 7 days of history, Pro keeps 1 year and Team "
     "keeps unlimited history."),
    ("trash-restore", "Recovering deleted notes",
     "Deleted notes go to the Trash and stay there for 30 days. To recover one, open Trash in the sidebar, select the "
     "note and click Restore. After 30 days notes are permanently deleted and cannot be recovered, even by support."),
    ("plans-pricing", "Plans and pricing",
     "Lumen Notes has three plans. Free: up to 3 devices and 100 MB of uploads per month. Pro: $6 per month billed "
     "yearly or $8 monthly, unlimited devices and 10 GB uploads. Team: $10 per user per month with shared notebooks, "
     "admin controls and SSO."),
    ("cancel-subscription", "Cancel your subscription",
     "You can cancel at any time from Settings > Billing > Cancel plan. You keep paid features until the end of the "
     "current billing period, then the account moves to the Free plan. Your notes are never deleted when you cancel."),
    ("refund-policy", "Refund policy",
     "Yearly Pro and Team subscriptions can be refunded in full within 14 days of purchase. Monthly subscriptions are "
     "not refunded, but you can cancel to stop the next charge. Request a refund through the Help Center contact form."),
    ("student-discount", "Student and education discount",
     "Students and teachers get 50% off Pro. Verify your status with a school email address under Settings > Billing > "
     "Education discount. Verification must be renewed every 12 months."),
    ("device-limit", "Device limit on the Free plan",
     "The Free plan can be used on up to 3 devices. If you sign in on a fourth device you will be asked to sign out of "
     "another one first. Manage devices under Settings > Account > Devices. Pro and Team have no device limit."),
    ("two-factor", "Two-factor authentication",
     "Turn on two-factor authentication (2FA) in Settings > Security. Lumen Notes supports authenticator apps and "
     "hardware security keys. Save your backup codes; if you lose your 2FA device and your backup codes, support must "
     "verify your identity before access is restored, which can take up to 5 business days."),
    ("reset-password", "Reset your password",
     "Click 'Forgot password' on the sign-in screen and enter your account email. The reset link is valid for 1 hour. "
     "If you don't receive it, check your spam folder or request a new link after 5 minutes."),
    ("encryption", "How your notes are encrypted",
     "Notes are encrypted in transit with TLS 1.3 and at rest with AES-256. Pro and Team users can also lock individual "
     "notes with a separate password; locked notes are end-to-end encrypted and Lumen staff cannot read them. If you "
     "forget a note password, that note cannot be recovered."),
    ("shared-notebooks", "Sharing notebooks with others",
     "Team plans can share notebooks with workspace members as Viewer, Commenter or Editor. Pro users can share a "
     "single note with a public read-only link. Public links can be disabled at any time from the Share menu."),
    ("web-clipper", "Web clipper browser extension",
     "The Lumen web clipper saves articles, screenshots or whole pages into a notebook. It is available for Chrome, "
     "Firefox, Edge and Safari. After installing, sign in from the extension icon and choose a default notebook."),
    ("templates", "Using note templates",
     "Templates give new notes a ready-made structure, such as meeting notes or a weekly review. Create a note from "
     "a template with New > From template. Save any note as your own template from the ... menu > Save as template."),
    ("tags-search", "Tags and search",
     "Add tags by typing # followed by a word inside a note. Search supports quoted phrases, tag:name filters and "
     "created:2024 style date filters. Pro and Team plans also search text inside images and PDF attachments."),
    ("ai-summary", "AI summaries",
     "Pro and Team plans include AI summaries. Select Summarize in the toolbar to get a short summary of a long note. "
     "Summaries are generated on demand and never used to train AI models. Workspace admins on Team can turn the "
     "feature off for everyone."),
    ("import-evernote", "Import from other apps",
     "Import notes from Evernote (.enex files), Notion (Markdown export), or plain Markdown and text files using "
     "Settings > Data > Import. Tags and creation dates are preserved where the source app provides them."),
    ("storage-limits", "Attachment size and storage limits",
     "A single attachment can be up to 25 MB on Free and 200 MB on Pro and Team. Monthly upload limits are 100 MB on "
     "Free and 10 GB on Pro. Team workspaces get 20 GB per user, pooled across the workspace."),
    ("delete-account", "Delete your account",
     "To permanently delete your account and all notes, go to Settings > Account > Delete account and confirm with "
     "your password. Deletion is processed within 30 days. Export your notes first; deleted data cannot be restored."),
    ("keyboard-shortcuts", "Keyboard shortcuts",
     "Common shortcuts: Cmd/Ctrl+N new note, Cmd/Ctrl+K quick switcher, Cmd/Ctrl+Shift+F search all notes, "
     "Cmd/Ctrl+/ show every shortcut. Shortcuts can be customised in Settings > Keyboard."),
    ("data-residency", "Where your data is stored",
     "By default, data is stored in data centers in the United States. Team workspaces can choose EU data residency "
     "when the workspace is created; it cannot be changed afterwards."),
]

# (question, relevant doc ids, reference answer, type)
QUESTIONS = [
    # keyword: shares vocabulary with the docs
    ("How do I export a note as PDF?", ["export-pdf"], "Open the note, click Share and choose Export as PDF.", "keyword"),
    ("Can I export my notes to Markdown?", ["export-markdown"], "Yes: Share > Export as Markdown; bulk export is on Pro and Team.", "keyword"),
    ("How do I resolve a sync conflict?", ["sync-conflicts"], "Both versions are kept; compare the 'Conflicted copy' with the original and delete the one you don't need.", "keyword"),
    ("How long is version history kept on the Free plan?", ["version-history"], "7 days.", "keyword"),
    ("How do I restore a deleted note from the Trash?", ["trash-restore"], "Open Trash, select the note and click Restore, within 30 days.", "keyword"),
    ("What does the Team plan cost per user?", ["plans-pricing"], "$10 per user per month.", "keyword"),
    ("How do I cancel my subscription?", ["cancel-subscription"], "Settings > Billing > Cancel plan.", "keyword"),
    ("What is the refund policy for yearly subscriptions?", ["refund-policy"], "Full refund within 14 days of purchase.", "keyword"),
    ("Is there a student discount?", ["student-discount"], "Yes, 50% off Pro, verified with a school email.", "keyword"),
    ("How do I turn on two-factor authentication?", ["two-factor"], "Settings > Security; authenticator apps and hardware keys are supported.", "keyword"),
    ("How long is the password reset link valid?", ["reset-password"], "1 hour.", "keyword"),
    ("Which browsers does the web clipper support?", ["web-clipper"], "Chrome, Firefox, Edge and Safari.", "keyword"),
    ("How do I create a note from a template?", ["templates"], "New > From template.", "keyword"),
    ("Can I import my Evernote notes?", ["import-evernote"], "Yes, import .enex files via Settings > Data > Import.", "keyword"),
    ("What is the maximum attachment size on Pro?", ["storage-limits"], "200 MB per attachment.", "keyword"),
    ("Can Team workspaces use EU data residency?", ["data-residency"], "Yes, chosen when the workspace is created; it can't be changed later.", "keyword"),
    ("What is the keyboard shortcut for the quick switcher?", ["keyboard-shortcuts"], "Cmd/Ctrl+K.", "keyword"),
    ("Do AI summaries use my notes to train models?", ["ai-summary"], "No, summaries are never used to train AI models.", "keyword"),
    # paraphrase: same intent, different words
    ("My laptop has no wifi on the plane, can I still write?", ["offline-mode"], "Yes, the desktop app works offline and syncs edits when you reconnect.", "paraphrase"),
    ("I edited the same page on my phone and my computer and now there are two of them, why?", ["sync-conflicts"], "It's a sync conflict; both versions are kept as the note and a 'Conflicted copy'.", "paraphrase"),
    ("I messed up a paragraph yesterday, can I go back to what it said before?", ["version-history"], "Yes, use version history (clock icon) to preview and restore an earlier version.", "paraphrase"),
    ("I accidentally binned something last week, is it gone forever?", ["trash-restore"], "No, it stays in Trash for 30 days and can be restored.", "paraphrase"),
    ("How much money is the paid tier if I pay once a year?", ["plans-pricing"], "Pro is $6 per month billed yearly.", "paraphrase"),
    ("If I stop paying, will I lose everything I wrote?", ["cancel-subscription"], "No, notes are never deleted; the account moves to Free at the end of the period.", "paraphrase"),
    ("I bought the annual plan by mistake three days ago, can I get my money back?", ["refund-policy"], "Yes, yearly plans are fully refundable within 14 days.", "paraphrase"),
    ("I'm at university, is it cheaper for me?", ["student-discount"], "Yes, students get 50% off Pro.", "paraphrase"),
    ("It won't let me log in on my new tablet, says too many devices", ["device-limit"], "Free allows 3 devices; sign out of one under Settings > Account > Devices or upgrade.", "paraphrase"),
    ("I lost my phone with the authenticator app, how do I get back in?", ["two-factor"], "Use your backup codes; otherwise support verifies your identity, which can take up to 5 business days.", "paraphrase"),
    ("Can your employees read my private journal?", ["encryption"], "Locked notes are end-to-end encrypted and staff cannot read them; all notes are encrypted in transit and at rest.", "paraphrase"),
    ("How do I let a coworker edit my project notes?", ["shared-notebooks"], "On Team, share the notebook with them as an Editor.", "paraphrase"),
    ("Can I save a news article from Chrome straight into my notes?", ["web-clipper"], "Yes, with the web clipper browser extension.", "paraphrase"),
    ("Can I find words that are inside a scanned receipt photo?", ["tags-search"], "Yes, on Pro and Team search covers text inside images.", "paraphrase"),
    ("Is there a way to get a quick TL;DR of a really long meeting write-up?", ["ai-summary"], "Yes, use Summarize (AI summaries) on Pro and Team.", "paraphrase"),
    ("I'm moving from Notion, can I bring my pages?", ["import-evernote"], "Yes, import Notion's Markdown export via Settings > Data > Import.", "paraphrase"),
    ("I want all my data wiped from your servers", ["delete-account"], "Delete the account in Settings > Account > Delete account; it's processed within 30 days.", "paraphrase"),
    ("Are my files kept in Europe?", ["data-residency"], "By default data is in the US; Team workspaces can choose EU residency at creation.", "paraphrase"),
    # unanswerable: nothing in the knowledge base answers these
    ("Does Lumen Notes have a Linux desktop app?", [], "Not covered by the knowledge base.", "unanswerable"),
    ("Can I pay with PayPal?", [], "Not covered by the knowledge base.", "unanswerable"),
    ("Is there an API for developers?", [], "Not covered by the knowledge base.", "unanswerable"),
    ("Can I record audio notes?", [], "Not covered by the knowledge base.", "unanswerable"),
]


def main() -> None:
    with open(HERE / "docs.jsonl", "w") as f:
        for doc_id, title, text in DOCS:
            f.write(json.dumps({"id": doc_id, "title": title, "text": text}) + "\n")
    ids = {d[0] for d in DOCS}
    with open(HERE / "golden_set.jsonl", "w") as f:
        for i, (q, rel, ref, qtype) in enumerate(QUESTIONS, 1):
            assert all(r in ids for r in rel), rel
            f.write(json.dumps({"id": f"q{i:02d}", "question": q, "relevant_ids": rel,
                                "reference_answer": ref, "type": qtype}) + "\n")
    print(f"{len(DOCS)} docs, {len(QUESTIONS)} questions")


if __name__ == "__main__":
    main()
