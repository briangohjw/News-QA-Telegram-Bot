from dbhelper import DBHelper
from newsqa import NewsQaModel, get_single_prediction
from newspaper import Article
from rank_bm25 import BM25Okapi
from string import punctuation
from telegram.ext import Updater
from telegram import  ReplyKeyboardMarkup, KeyboardButton, ParseMode
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    Filters
)
from transformers import BertTokenizer, BertForQuestionAnswering
import json
import logging
import re
import spacy
import torch

# News Processing Methods
def get_article_content(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def get_static_content(content):
    # Adds full article content to the static_content json
    articles = content["articles"]
    for article in articles:
        url = article["url"]
        text = get_article_content(url)
        article["full_content"] = text
    return content

def setup_static_content(filename):
    f = open(filename,)
    data = json.load(f)
    content = get_static_content(data)
    json_object = json.dumps(content, indent = 4)

    # Writing to sample.json
    with open("static_content.json", "w") as outfile:
        outfile.write(json_object)

def get_headlines(content):
    headlines = []
    articles = content["articles"]
    for article in articles:
        headline = article["title"]
        url = article['url']
        url_parsed = f"<a href='{url}'>(Link)</a>"
        headlines.append((headline, url_parsed))
    return format_headlines(headlines)

def get_full_contents(content):
    full_contents = []
    articles = content["articles"]
    for article in articles:
        full_content = article["full_content"].lower()
        full_contents.append(full_content)
    return full_contents

# Telebot Handlers
def start(update, context):
    response = "1. Choose /send_text to submit your own text via copy-paste 📝"
    response += "\n\n2. Choose /send_url to submit your own text via url 🌎"
    response += "\n\n3. Choose /headlines to read the headlines ❗️"

    keyboard = [
        [
            KeyboardButton("/send_text"),
            KeyboardButton("/send_url"),
        ],
        [KeyboardButton("/headlines")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)

    update.message.reply_text(response, reply_markup=reply_markup)

def text(update, context):
    response = "Send me a story"
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    return TEXT

def get_story_from_text(update, context):
    user_id = str(update.effective_user.id)
    story = update.message.text
    db.add_story(user_id, story)
    response = "Ask me any questions regarding the text"
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    return ConversationHandler.END

def url(update, context):
    response = "Send me a URL to your story"
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    return TEXT

def get_story_from_url(update, context):
    user_id = str(update.effective_user.id)
    url = update.message.text
    story = get_article_content(url)
    db.add_story(user_id, story)
    response = "Ask me any questions regarding the text"
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    return ConversationHandler.END

def headlines(update, context):
    f = open('static_content.json',)
    data = json.load(f)
    headers = get_headlines(data)
    context.bot.send_message(chat_id=update.effective_chat.id, text=headers, parse_mode=ParseMode.HTML)
    response = "🙋🏻‍♂️ Ask a question regarding any of the headlines"
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    return TEXT

def handle_ama_question(update, context):
    question = update.message.text
    response = "<b>You asked me:</b> " + question
    user_id = update.effective_user.id
    db.add_question(user_id, question)
    print(f"adding user_id: {user_id}")
    print(f"adding question: {question}")
    context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode=ParseMode.HTML)
    # perform IR
    retrieved_document = bm25.get_top_n(question.lower().split(), bm25_corpus, n=1)[0]
    ans = qa(retrieved_document, question)
    # print(ans)
    if not ans:
        no_response = "I do not know"
        context.bot.send_message(chat_id=update.effective_chat.id, text=no_response)
    else:
        answer = ans[0].split('\n')[0].strip().strip(punctuation)
        response = f"<b>Answer:</b> {answer}"
        context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode=ParseMode.HTML)
    return ConversationHandler.END

def handle_followup_question(update, context):
    question = ' '.join(context.args).strip()
    # perform query reformulation
    user_id = update.effective_user.id
    prev_question = db.get_question(user_id)[0]
    print(f"prev_qn: {prev_question}")
    if prev_question:
        this_pos_tagged = spacy_pos_tag(question)
        prev_ner_tagged = spacy_ner_tag(prev_question)
        prev_persons = get_per_ners(prev_ner_tagged)
        candidate_questions = replace_pronouns(this_pos_tagged, prev_persons)
        if candidate_questions:
            question = candidate_questions[0]
    else:
        context.bot.send_message(chat_id=update.effective_chat.id, text="You did not ask any previous questions")
        return ConversationHandler.END
    response = "<b>You asked me:</b> " + question
    context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode=ParseMode.HTML)
    # perform IR
    retrieved_document = bm25.get_top_n(question.lower().split(), bm25_corpus, n=1)[0]
    ans = qa(retrieved_document, question)
    # print(ans)
    if not ans:
        no_response = "I do not know"
        context.bot.send_message(chat_id=update.effective_chat.id, text=no_response)
    else:
        answer = ans[0].split('\n')[0].strip().strip(punctuation)
        response = f"<b>Answer:</b> {answer}"
        context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode=ParseMode.HTML)
    return ConversationHandler.END

def format_headlines(headlines):
    formatted_headlines = '❗️<b>Headlines:</b>❗️\n\n'
    for i, data in enumerate(headlines):
        headline, url_parsed = data
        if headline.endswith('.com'):
            headline = ' '.join(headline.split()[:-1])
        formatted_headlines += f"{i+1}. {headline} {url_parsed}\n\n"
    return formatted_headlines

# When the user just sends a message/qn, without calling any prior command
# The bot will check if there is any stored stories, and prompt for a story if absent
def question(update, context):
    user_id = update.effective_user.id
    story = db.get_story(user_id)
    question = update.message.text
    if not story:
            response = "1. Type /send_text to submit your own text via copy-paste 📝"
            response += "\n\n2. Type /send_url to submit your own text via url 🌎"
            response += "\n\n3. Type /headlines to read the headlines ❗️"
            response += "\n\n4. Type /followup [question] to ask a follow-up question 🧠"
            response += "\n\n👍"
            context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    else:
        story = story[0]
        ans = qa(story, question)
        # print(ans)
        if not ans:
            no_response = "I do not know"
            context.bot.send_message(chat_id=update.effective_chat.id, text=no_response)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text=ans[0])

# QA methods
def qa(story,question):
    # print("story: " + story)
    # print("question: " + question)
    ans = get_single_prediction(text=story, question=question, tokenizer=tokenizer, newsqa_model=bert_model)
    filtered_ans = trim_answers(ans)
    return filtered_ans

def trim_answers(answers):
    # Some answers are blank
    answers = answers[0]
    filtered = []
    for ans in answers:
        if len(ans) != 0:
            filtered.append(ans)
    return filtered

def spacy_pos_tag(text):
    result = []
    doc = nlp(text)
    for token in doc:
        result.append((token.text, token.tag_))
    return result

def spacy_ner_tag(text):
    result = []
    doc = nlp(text)
    for i in doc:
        result.append((i.text, i.ent_type_, i.ent_iob_))
    return result

def get_per_ners(ner_tagged):
  persons = []
  prev_flag = False
  current_per = None

  for word, tag, bio in ner_tagged:
    if tag == 'PERSON' and bio == 'B': # new person
      current_per = [word]
    elif tag == 'PERSON' and bio in ('I', 'O'): # part of same person
      current_per.append(word)
    else: # tag is not PER. append current person to result (if any)
      if current_per is not None:
        if len(current_per) == 1:
          persons.extend(current_per)
        else:
          persons.append(' '.join(current_per))
        current_per = None
  if current_per:
    persons.extend(current_per)
  return persons

def replace_pronouns(pos_tagged, prev_persons):
  result = []
  this_pronoun_list = []
  words = []

  for word, tag in pos_tagged:
    if tag in ('PRP', 'PRP$'):
      this_pronoun_list.append(word)
    words.append(word)

  for noun in prev_persons:
    this_token_list = []
    for word in words:
      if word in this_pronoun_list:
        this_token_list.append(noun)
      else:
        this_token_list.append(word)
    result.append(re.sub('\s+[?]', '?', ' '.join(this_token_list)))

  return result

def main():
    updater = Updater(token=TOKEN, use_context=True)

    user_submitted_text_handler = ConversationHandler(
        entry_points=[CommandHandler('send_text', text)],
        states={
            TEXT: [MessageHandler(Filters.text, get_story_from_text)],
        },
        fallbacks=[],
    )

    user_submitted_url_handler = ConversationHandler(
        entry_points=[CommandHandler('send_url', url)],
        states={
            TEXT: [MessageHandler(Filters.text, get_story_from_url)],
        },
        fallbacks=[],
    )

    headlines_handler = ConversationHandler(
        entry_points=[CommandHandler('headlines', headlines)],
        states={
            TEXT: [MessageHandler(Filters.text, handle_ama_question)]
        },
        fallbacks=[],
    )

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(user_submitted_text_handler)
    updater.dispatcher.add_handler(user_submitted_url_handler)
    updater.dispatcher.add_handler(headlines_handler)
    updater.dispatcher.add_handler(CommandHandler('followup', handle_followup_question))
    updater.dispatcher.add_handler(MessageHandler(
        filters=Filters.text,
        callback=question)
    )

    updater.start_polling()

# Setup BM25 IR
f = open('static_content.json',)
data = json.load(f)
full_contents = get_full_contents(data)
bm25_corpus = list(full_contents)
tokenized_bm25corpus = [doc.split(" ") for doc in bm25_corpus]
bm25 = BM25Okapi(tokenized_bm25corpus)

# Setup BERT Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Option 1: use non-finetuned
# bert_non_finetuned = BertForQuestionAnswering.from_pretrained(bert_model_name)
# bert_non_finetuned.to(device)
# bert_model = NewsQaModel(bert_non_finetuned)

# Option2: use fine-tuned, requires downloading bert_sample.pt from GDrive
bert_model = NewsQaModel()
bert_model.load('bert_sample.pt')

TOKEN = "INSERT_TOKEN_HERE"

db = DBHelper()

nlp = spacy.load("en_core_web_sm")

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

TEXT = range(1)
URL = range(1)

if __name__ == "__main__":
    # setup_static_content('static_content.json')
    db.setup()
    main()