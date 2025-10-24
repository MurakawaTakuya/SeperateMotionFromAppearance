"""
CLIP類似度計算ユーティリティ
Target VerbとVerb Dictionary間の類似度を計算し、softmax正規化された重みを返す
"""

import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import CLIPModel, CLIPProcessor


class ClipSimilarityCalculator:
    """CLIPを使ったテキスト間類似度計算クラス"""

    def __init__(self, clip_model: CLIPModel = None, device: str = "cuda", model_name: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            clip_model: CLIPモデル（Noneの場合は指定されたモデル名から読み込み）
            device: 計算デバイス
            model_name: 使用するCLIPモデルの名前
        """
        self.device = device
        self.model_name = model_name

        if clip_model is not None:
            self.clip_model = clip_model
            self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            # Hugging FaceのCLIPモデルを読み込み
            self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        # モデルを評価モードに設定
        self.clip_model.eval()

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        複数のテキストをCLIPでエンコードする

        Args:
            texts: エンコードするテキストのリスト

        Returns:
            torch.Tensor: エンコードされたテキスト埋め込み (batch_size, embedding_dim)
        """
        # CLIPプロセッサーを使用してテキストを処理
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)

        # デバイスに移動
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # CLIPモデルでテキストをエンコード
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        # L2正規化（CLIPの標準的な方法）
        text_features = F.normalize(text_features, p=2, dim=1)

        return text_features

    def calculate_similarities(self, target_verb: str, verb_dictionary: List[str], temperature: float = 1.0) -> Dict[str, float]:
        """
        Target VerbとVerb Dictionary間の類似度を計算し、softmax正規化された重みを返す

        Args:
            target_verb: ターゲット動詞
            verb_dictionary: 動詞辞書のリスト
            temperature: softmaxの温度パラメータ（高いほど分布が均一になる）

        Returns:
            Dict[str, float]: 各動詞に対する重みの辞書
        """
        # 全テキストを結合（target_verb + verb_dictionary）
        all_texts = [target_verb] + verb_dictionary

        # 全テキストをエンコード
        embeddings = self.encode_texts(all_texts)

        # target_verbの埋め込み
        target_embedding = embeddings[0:1]  # (1, embedding_dim)

        # verb_dictionaryの埋め込み
        verb_embeddings = embeddings[1:]  # (len(verb_dictionary), embedding_dim)

        # コサイン類似度を計算
        # target_embeddingと各verb_embeddingの内積（正規化済みなのでコサイン類似度）
        similarities = torch.mm(target_embedding, verb_embeddings.T).squeeze(0)  # (len(verb_dictionary),)

        # 温度パラメータでスケーリング
        scaled_similarities = similarities / temperature

        # softmax正規化
        weights = F.softmax(scaled_similarities, dim=0)

        # 辞書形式で返す
        result = {}
        for i, verb in enumerate(verb_dictionary):
            result[verb] = weights[i].item()

        return result

    def calculate_similarities_batch(self, target_verbs: List[str], verb_dictionary: List[str], temperature: float = 1.0) -> List[Dict[str, float]]:
        """
        複数のTarget Verbに対して一括で類似度を計算する

        Args:
            target_verbs: ターゲット動詞のリスト
            verb_dictionary: 動詞辞書のリスト
            temperature: softmaxの温度パラメータ

        Returns:
            List[Dict[str, float]]: 各Target Verbに対する重みの辞書のリスト
        """
        results = []
        for target_verb in target_verbs:
            weights = self.calculate_similarities(target_verb, verb_dictionary, temperature)
            results.append(weights)
        return results


def main():
    """CLIP類似度計算のメイン関数"""
    parser = argparse.ArgumentParser(description="CLIP類似度計算")
    parser.add_argument("--target_verb", type=str,
                        default="human action of high jump",
                        help="Target Verb")
    # parser.add_argument("--verb_dictionary", nargs="+",
    #                     default=["cartwheel", "dive", "handstand", "jump", "run",
    #                              "somersault", "stand", "throw", "turn", "walk"],
    #                     help="Verb Dictionary list")
    parser.add_argument("--verb_dictionary", nargs="+",
                        default=["human action of cartwheeling", "human action of diving", "human action of handstanding", "human action of jumping", "human action of running",
                                 "human action of somersaulting", "human action of standing", "human action of throwing", "human action of turning", "human action of walking"],
                        help="Verb Dictionary list")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature parameter for softmax")
    parser.add_argument("--device", type=str, default="auto",
                        help="Calculation device (auto, cuda, cpu)")
    parser.add_argument("--clip_model_name", type=str,
                        default="openai/clip-vit-base-patch32",
                        help="Name of the CLIP model to use")

    args = parser.parse_args()

    # デバイス設定
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"CLIPモデル: {args.clip_model_name}")
    print(f"Target Verb: '{args.target_verb}'")
    print(f"Verb Dictionary: {args.verb_dictionary}")
    print(f"Temperature: {args.temperature}")

    try:
        clip = ClipSimilarityCalculator(device=device, model_name=args.clip_model_name)

        # 類似度を計算
        weights = clip.calculate_similarities(
            args.target_verb,
            args.verb_dictionary,
            args.temperature
        )

        # 重みを降順でソート
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        print("result:")
        for verb, weight in sorted_weights:
            print(f"{verb:12s}: {weight:.4f}")

        print()
        print("sum:", sum(weights.values()))
        print("max:", max(weights.values()))
        print("min:", min(weights.values()))

    except FileNotFoundError as e:
        print(f"エラー: CLIPモデルが見つかりません: {e}")
        print("--clip_model_nameで正しいモデル名を指定してください")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
