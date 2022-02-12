import os
import text_generation

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    FollowEvent, MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackTemplateAction, MessageTemplateAction, URITemplateAction
)

# 軽量なウェブアプリケーションフレームワーク:Flask
app = Flask(__name__)

#環境変数からLINE Access Tokenを設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
#環境変数からLINE Channel Secretを設定
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# LINEMessagingAPI との接続
@app.route("/callback", methods=['POST'])
def callback():
    # get X‐Line‐Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channelsecret.")
        abort(400)
    return "OK"

# 応答のプログラム
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """
    LINEへのテキストメッセージに対して応答を返す

    Parameters
    ----------
    event: MessageEvent
      LINEに送信されたメッセージイベント
    """
    # 100語の文字列が返ってくる
    reply_message = text_generation.generation(event.message.text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="応答です。 " + reply_message))
	
# Flask 実行確認用
@app.route("/good")
def good():
	return "Hello Heroku"

if __name__ == "__main__":
    port = int(os.getenv("PORT"))
    app.run(host="0.0.0.0", port=port)