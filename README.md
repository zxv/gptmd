# gptmd: ChatGPT Export to Markdown

Turn your ChatGPT export into readable Markdown files with one script.

## Quick start

1. Clone the repo:
   ```
   git clone https://github.com/zxv/gptmd.git
   cd gptmd
   ```
2. Get your ChatGPT export (ChatGPT Settings → Data controls → Export data), put the zip in the `data/` folder, and extract it there. If you've already extracted it, move those JSON files to that folder.

   Now you are set up to interact with your exported ChatGPT data.

## Possible Actions

1. Run a quick summary:
   ```
   python3 gptmd.py -s
   ```
2. List conversation titles:
   ```
   python3 gptmd.py -l
   ```
3. Export everything to Markdown (goes to `output/` by default):
   ```
   python3 gptmd.py -e
   ```

## What you get

- One `.md` file per conversation, named like `YYYY-MM-DD - Title.md`.
- Cleaned text with working links (citations are resolved).
- All conversation branches included by default so you can see alternate replies.
- Optional extras:
  - Include reasoning/thought messages: add `-t`
  - Include tool-call JSON blocks: add `-T`
  - Use UTC timestamps: add `-u`
  - Choose a different output folder: `-o myfolder`
  - Point to a different data folder: `-d path/to/data`

## Notes

- You'll need to place your exported `conversations.json` in `data/` before running.
- Empty messages are skipped from the export; thoughts and tool calls stay hidden unless you ask for them.
- Branching conversations are shown inline under “Alternate branch …” headings so it’s clear where replies diverged.

## Repo layout

- `gptmd.py` — the script.
- `data/` — put your `conversations.json` export here (placeholder file included).
- `output/` — Markdown files are written here (placeholder file included).
