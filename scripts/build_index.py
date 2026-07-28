"""为诗文正文和赏析构建本地 Chroma 向量索引。

正文与每一段赏析分别 embedding，并写入两个 collection。脚本可重复运行：
全部文本成功编码且向量维度校验通过后，才删除并重建旧 collection。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "poems.json"
DEFAULT_CHROMA_PATH = ROOT / "chroma"
DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
EXPECTED_DIMENSION = 1024
CONTENT_COLLECTION = "poem_content"
APPRECIATION_COLLECTION = "poem_appreciation"


@dataclass(frozen=True)
class VectorRecord:
    """一条待写入 Chroma 的记录。"""

    record_id: str
    document: str
    metadata: dict[str, str]


def load_records(data_path: Path) -> tuple[list[VectorRecord], list[VectorRecord]]:
    """读取并校验 poems.json，拆成正文记录和赏析记录。"""
    with data_path.open(encoding="utf-8") as file:
        poems = json.load(file)
    if not isinstance(poems, list):
        raise ValueError("poems.json 顶层必须是数组")

    content_records: list[VectorRecord] = []
    appreciation_records: list[VectorRecord] = []
    seen_ids: set[str] = set()

    for poem_index, poem in enumerate(poems):
        if not isinstance(poem, dict):
            raise ValueError(f"第 {poem_index} 首诗不是对象")

        fields = {}
        for key in ("poem_id", "title", "author", "dynasty", "content"):
            value = poem.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"第 {poem_index} 首诗的 {key} 缺失或为空")
            fields[key] = value

        poem_id = fields["poem_id"]
        if poem_id in seen_ids:
            raise ValueError(f"重复的 poem_id: {poem_id}")
        seen_ids.add(poem_id)

        common_metadata = {
            "poem_id": poem_id,
            "title": fields["title"],
            "author": fields["author"],
            "dynasty": fields["dynasty"],
        }
        content_records.append(
            VectorRecord(
                record_id=poem_id,
                # 正文原文是被 embedding 的文本，也作为 document 保存。
                document=fields["content"],
                metadata={
                    **common_metadata,
                    "kind": "content",
                    "evidence_id": "",
                },
            )
        )

        appreciation = poem.get("appreciation")
        if not isinstance(appreciation, list):
            raise ValueError(f"诗 {poem_id} 的 appreciation 必须是数组")
        seen_evidence_ids: set[str] = set()
        for segment_index, segment in enumerate(appreciation):
            if not isinstance(segment, dict):
                raise ValueError(f"诗 {poem_id} 的赏析段 {segment_index} 不是对象")
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"诗 {poem_id} 的赏析段 {segment_index} 文本为空")

            evidence_id = _short_evidence_id(
                segment.get("evidence_id"), poem_id, segment_index
            )
            if evidence_id in seen_evidence_ids:
                raise ValueError(f"诗 {poem_id} 的 evidence_id 重复: {evidence_id}")
            seen_evidence_ids.add(evidence_id)

            appreciation_records.append(
                VectorRecord(
                    record_id=f"{poem_id}#{evidence_id}",
                    # 每一段赏析独立 embedding，并保留原文以供检索结果取回。
                    document=text,
                    metadata={
                        **common_metadata,
                        "kind": "appreciation",
                        "evidence_id": evidence_id,
                    },
                )
            )

    return content_records, appreciation_records


def _short_evidence_id(raw_id: Any, poem_id: str, segment_index: int) -> str:
    """把数据中的 ``poem_id#appr-N`` 转成契约要求的短 ID。"""
    fallback = f"appr-{segment_index}"
    if raw_id is None:
        return fallback
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError(f"诗 {poem_id} 的赏析段 {segment_index} evidence_id 无效")

    prefix = f"{poem_id}#"
    short_id = raw_id[len(prefix) :] if raw_id.startswith(prefix) else raw_id
    if "#" in short_id or not short_id:
        raise ValueError(f"诗 {poem_id} 的 evidence_id 格式无效: {raw_id}")
    return short_id


def encode_records(
    model: Any,
    records: Sequence[VectorRecord],
    *,
    batch_size: int,
    label: str,
) -> Any:
    """批量编码，并在 encode 时开启 L2 归一化。"""
    documents = [record.document for record in records]
    print(f"编码{label}: {len(documents)} 条（batch_size={batch_size}）")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,  # BGE 推荐的 L2 归一化在这里完成。
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    if len(embeddings) != len(records):
        raise RuntimeError(
            f"{label}向量数异常: 预期 {len(records)}，实际 {len(embeddings)}"
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_DIMENSION:
        actual = embeddings.shape[1] if embeddings.ndim == 2 else embeddings.shape
        raise RuntimeError(
            f"{label}向量维度应为 {EXPECTED_DIMENSION}，实际为 {actual}"
        )
    return embeddings


def replace_collections(
    chroma_path: Path,
    content_records: Sequence[VectorRecord],
    content_embeddings: Any,
    appreciation_records: Sequence[VectorRecord],
    appreciation_embeddings: Any,
    *,
    write_batch_size: int,
) -> None:
    """清空旧 collection，然后分批写入新索引。"""
    import chromadb

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    for name in (CONTENT_COLLECTION, APPRECIATION_COLLECTION):
        if name in existing_names:
            client.delete_collection(name=name)

    # 向量已 L2 归一化；cosine 距离也明确记录在 collection 配置中。
    content_collection = client.create_collection(
        name=CONTENT_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    appreciation_collection = client.create_collection(
        name=APPRECIATION_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    _add_in_batches(
        content_collection,
        content_records,
        content_embeddings,
        write_batch_size,
        "正文",
    )
    _add_in_batches(
        appreciation_collection,
        appreciation_records,
        appreciation_embeddings,
        write_batch_size,
        "赏析",
    )

    if content_collection.count() != len(content_records):
        raise RuntimeError("正文 collection 写入后的记录数不一致")
    if appreciation_collection.count() != len(appreciation_records):
        raise RuntimeError("赏析 collection 写入后的记录数不一致")


def _add_in_batches(
    collection: Any,
    records: Sequence[VectorRecord],
    embeddings: Any,
    batch_size: int,
    label: str,
) -> None:
    total = len(records)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = records[start:end]
        collection.add(
            ids=[record.record_id for record in batch],
            documents=[record.document for record in batch],
            metadatas=[record.metadata for record in batch],
            embeddings=embeddings[start:end].tolist(),
        )
        print(f"写入{label}: {end}/{total}", end="\r" if end < total else "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建诗文 BGE/Chroma 向量索引")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer 模型名或本地模型目录",
    )
    parser.add_argument("--device", help="例如 cpu、cuda 或 mps；默认自动选择")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--write-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.write_batch_size <= 0:
        raise SystemExit("batch size 必须大于 0")

    started_at = time.perf_counter()
    content_records, appreciation_records = load_records(args.data.resolve())
    print(
        f"已读取 {len(content_records)} 首诗，"
        f"共 {len(appreciation_records)} 段赏析"
    )

    from sentence_transformers import SentenceTransformer

    print(f"加载模型: {args.model}")
    model_kwargs = {"device": args.device} if args.device else {}
    model = SentenceTransformer(args.model, **model_kwargs)

    content_embeddings = encode_records(
        model, content_records, batch_size=args.batch_size, label="正文"
    )
    appreciation_embeddings = encode_records(
        model, appreciation_records, batch_size=args.batch_size, label="赏析"
    )
    replace_collections(
        args.chroma_dir.resolve(),
        content_records,
        content_embeddings,
        appreciation_records,
        appreciation_embeddings,
        write_batch_size=args.write_batch_size,
    )

    elapsed = time.perf_counter() - started_at
    print("\n索引构建完成")
    print(f"正文向量数: {len(content_records)}")
    print(f"赏析向量数: {len(appreciation_records)}")
    print(f"向量维度: {EXPECTED_DIMENSION}")
    print(f"Chroma 目录: {args.chroma_dir.resolve()}")
    print(f"耗时: {elapsed:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
