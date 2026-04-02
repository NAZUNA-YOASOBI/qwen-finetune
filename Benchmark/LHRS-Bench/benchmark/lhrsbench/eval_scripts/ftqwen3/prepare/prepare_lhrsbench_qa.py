from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = (
    "Please answer the multiple-choice question based on the given image and choices:\n"
    "Question: {question}\n"
    "Choice: {choices}\n"
    "Return only one capital letter from the given choices. Do not explain. "
    "Do not describe the image. Do not output the option text.\n"
    "Answer:"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _parse_choice_items(choice_text: str) -> list[tuple[str, str]]:
    parts = [x.strip() for x in str(choice_text).split(";") if x.strip()]
    out: list[tuple[str, str]] = []
    for part in parts:
        match = re.match(r"^([A-Z])\.\s*(.+)$", part)
        if not match:
            raise ValueError(f"Invalid choice item format: {part}")
        out.append((match.group(1).upper(), match.group(2).strip()))
    if len(out) < 2:
        raise ValueError(f"Need at least 2 choices, got: {choice_text}")
    return out


def _answer_letter(answer: str) -> str:
    match = re.search(r"([A-Z])", str(answer).upper())
    if not match:
        raise ValueError(f"Invalid answer format: {answer}")
    return match.group(1)


def _build_choices_text(items: list[tuple[str, str]]) -> str:
    return "; ".join(f"{letter}. {text}" for letter, text in items)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LHRS-Bench QA files with the paper-aligned 4-repeat setup.")
    parser.add_argument("--dataset-root", type=str, default="../../../LHRS-Bench/datasets/LHRS-Bench")
    parser.add_argument("--output-dir", type=str, default="benchmark/lhrsbench/data")
    parser.add_argument("--num-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if int(args.num_repeats) <= 0:
        raise ValueError("num_repeats must be > 0")

    project_root = _project_root()
    dataset_root = _resolve_from_project(args.dataset_root)
    output_dir = _resolve_from_project(args.output_dir)

    qa_path = dataset_root / "qa.json"
    img_png_dir = dataset_root / "imgs_png"
    img_jpg_dir = dataset_root / "imgs_jpg"

    if not qa_path.is_file():
        raise FileNotFoundError(f"Missing qa.json: {qa_path}")

    qa_obj = json.loads(qa_path.read_text(encoding="utf-8"))
    qtype = dict(qa_obj.get("qtype", {}))

    id_to_type_name: dict[str, str] = {}
    for key in qtype.keys():
        match = re.match(r"^\s*(\d+)\s+(.+?)\s*$", str(key))
        if not match:
            continue
        id_to_type_name[str(int(match.group(1)))] = str(match.group(2)).strip()

    questions: list[dict[str, Any]] = []
    qid = 0

    for image_idx, item in enumerate(qa_obj.get("data", [])):
        filename = str(item.get("filename", "")).strip()
        if not filename:
            raise ValueError(f"Empty filename at image_idx={image_idx}")

        image_path_png = img_png_dir / filename
        if image_path_png.is_file():
            image_path = image_path_png
        else:
            stem = Path(filename).stem
            image_path_jpg = img_jpg_dir / f"{stem}.jpg"
            if image_path_jpg.is_file():
                image_path = image_path_jpg
            else:
                raise FileNotFoundError(f"Missing image for {filename}: neither {image_path_png} nor {image_path_jpg}")

        image_path_rel = os.path.relpath(image_path, project_root)

        for qa_pair in item.get("qa_pairs", []):
            question = str(qa_pair.get("question", "")).strip()
            choices_raw = str(qa_pair.get("choices", "")).strip()
            answer_raw = str(qa_pair.get("answer", "")).strip()
            type_ids = [str(int(type_id)) for type_id in qa_pair.get("type", [])]
            if not question or not choices_raw or not answer_raw:
                raise ValueError(f"Invalid qa pair under {filename}: {qa_pair}")

            choice_items = _parse_choice_items(choices_raw)
            answer_letter = _answer_letter(answer_raw)
            choice_map = {letter: text for letter, text in choice_items}
            if answer_letter not in choice_map:
                raise ValueError(f"Answer {answer_letter} not found in choices for qid={qid}: {choices_raw}")
            answer_text = choice_map[answer_letter]

            questions.append(
                {
                    "qid": int(qid),
                    "filename": filename,
                    "image_path": str(image_path_rel),
                    "question": question,
                    "choices_raw": _build_choices_text(choice_items),
                    "answer_letter_raw": answer_letter,
                    "answer_text_raw": answer_text,
                    "type_ids": type_ids,
                    "type_names": [id_to_type_name.get(type_id, f"type_{type_id}") for type_id in type_ids],
                }
            )
            qid += 1

    attempts: list[dict[str, Any]] = []
    for question_row in questions:
        raw_items = _parse_choice_items(str(question_row["choices_raw"]))
        answer_text_raw = str(question_row["answer_text_raw"])

        for attempt_idx in range(int(args.num_repeats)):
            rng = random.Random(int(args.seed) + int(question_row["qid"]) * 10007 + int(attempt_idx) * 7919)
            shuffled = list(raw_items)
            rng.shuffle(shuffled)

            new_items: list[tuple[str, str]] = []
            for index, (_, option_text) in enumerate(shuffled):
                new_letter = chr(ord("A") + index)
                new_items.append((new_letter, option_text))

            answer_letter_new = ""
            for letter, option_text in new_items:
                if option_text == answer_text_raw:
                    answer_letter_new = letter
                    break
            if not answer_letter_new:
                raise RuntimeError(
                    f"Cannot locate shuffled answer letter for qid={question_row['qid']} attempt={attempt_idx}"
                )

            choices_shuffled = _build_choices_text(new_items)
            prompt = PROMPT_TEMPLATE.format(question=str(question_row["question"]), choices=choices_shuffled)

            attempts.append(
                {
                    "uid": f"q{int(question_row['qid']):04d}_a{int(attempt_idx)}",
                    "qid": int(question_row["qid"]),
                    "attempt": int(attempt_idx),
                    "filename": str(question_row["filename"]),
                    "image_path": str(question_row["image_path"]),
                    "question": str(question_row["question"]),
                    "choices_shuffled": choices_shuffled,
                    "options": [{"letter": letter, "text": text} for letter, text in new_items],
                    "answer_letter": answer_letter_new,
                    "answer_text": answer_text_raw,
                    "type_ids": list(question_row["type_ids"]),
                    "type_names": list(question_row["type_names"]),
                    "prompt": prompt,
                    "num_choices": int(len(new_items)),
                }
            )

    questions_out = output_dir / "lhrsbench_questions.jsonl"
    attempts_out = output_dir / f"lhrsbench_attempts_r{int(args.num_repeats)}_seed{int(args.seed)}.jsonl"
    meta_out = output_dir / "lhrsbench_meta.json"

    _write_jsonl(questions_out, questions)
    _write_jsonl(attempts_out, attempts)

    meta = {
        "dataset_root": str(dataset_root),
        "qa_path": str(qa_path),
        "num_images": int(qa_obj.get("img_num", 0)),
        "num_questions": int(len(questions)),
        "num_attempts": int(len(attempts)),
        "num_repeats": int(args.num_repeats),
        "seed": int(args.seed),
        "qtype": qtype,
        "id_to_type_name": id_to_type_name,
        "prompt_template": PROMPT_TEMPLATE,
        "questions_jsonl": str(questions_out),
        "attempts_jsonl": str(attempts_out),
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] questions={len(questions)} attempts={len(attempts)}")
    print(f"[OK] wrote: {questions_out}")
    print(f"[OK] wrote: {attempts_out}")
    print(f"[OK] wrote: {meta_out}")


if __name__ == "__main__":
    main()
