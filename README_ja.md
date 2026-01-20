Master Duel Deck Recognizer
===
<table>
	<thead>
		<tr>
			<th style="text-align:center"><a href="README.md">English</a></th>
			<th style="text-align:center">日本語</th>
		</tr>
	</thead>
</table>

## 🚀 概要

このリポジトリは遊戯王のデッキ画像をカードリストに転換するツールです。
デッキ画像をアップロードし、詳細なカードリスト、遊戯王ニューロンにおけるデッキコードを入手することができます。

ローカル環境にWebサービスとしてデプロイ可能で、画像アップロードと結果表示のブラウザベースのインターフェースを提供します。
Discordボットなどの外部クライアントにおいて画像送信でデッキのデータを取得できるようにするAPIも公開しています。

## ✏️ TODO

- [ ] Generate Neuron Deck IDs

- [ ] Enhance Pendulum Monster recognition

- [ ] Optimize recognition for 60-card decks

- [x] Alternate Art recognition

- [x] Implement language selection for the webpage

## 🛠️ セットアップ

### 1️⃣ リポジトリをクローン

```bash
git clone https://github.com/BernieTv/ElevenLabs-Clone.git
```

### 2️⃣ ディレクトリへ移動

```bash
cd img2DeckCode
```

### 3️⃣ Pythonをインストール 🐍

Python 3.10をインストールしてください。インストール方法の公式サイトを参照してください。  
👉 [公式サイト](https://www.python.org/downloads/)

## 📦 パッケージをインストール

仮想環境(venv)の構築がおすすめです。
```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## ▶️ 実行

```bash
python ./main.py
```

```bash
cd web
./ngrok.exe http 8000
```

生成されたngrok URLでWEBインターフェースへアクセスしたりAPIコールを送ったりできます。