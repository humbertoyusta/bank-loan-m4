from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, CallbackContext, filters, CallbackQueryHandler
import pickle
from dotenv import load_dotenv
import os
import pandas as pd

# States
AGE, WORK_EXP, INCOME, FAMILY_MEMBERS, EDUCATION, SECURITIES_ACCOUNT, MORTGAGE, CD_ACCOUNT, ONLINE_BANKING, CREDIT_CARD, AVG_SPENDING = range(11)

async def start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('What is your age?')
    return AGE

async def age(update: Update, context: CallbackContext) -> int:
    user_age = update.message.text
    context.user_data['Age'] = int(user_age)
    await update.message.reply_text('How many years of work experience do you have?')
    return WORK_EXP

async def work_experience(update: Update, context: CallbackContext) -> int:
    user_work_exp = update.message.text
    context.user_data['Experience'] = int(user_work_exp)
    await update.message.reply_text('What is your yearly income? (In thousands of dollars)')
    return INCOME

async def income(update: Update, context: CallbackContext) -> int:
    user_income = update.message.text
    context.user_data['Income'] = int(user_income)
    await update.message.reply_text('With how many family members do you live? (Including yourself)')
    return FAMILY_MEMBERS

async def family_members(update: Update, context: CallbackContext) -> int:
    user_family_members = update.message.text
    context.user_data['Family'] = int(user_family_members)

    reply_keyboard = [['Undergraduate', 'Graduated', 'Advanced/Postgraduated']]
    await update.message.reply_text(
        'What is your level of education?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return EDUCATION

async def education(update: Update, context: CallbackContext) -> int:
    user_education = update.message.text
    education_map = {'Undergraduate': 1, 'Graduated': 2, 'Advanced/Postgraduated': 3}
    context.user_data['Education'] = education_map.get(user_education, 1)
    reply_keyboard = [['Yes', 'No']]
    await update.message.reply_text(
        'Do you have a Securities Account with the bank?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return SECURITIES_ACCOUNT

async def securities(update: Update, context: CallbackContext) -> int:
    context.user_data['Securities Account'] = int(update.message.text.lower() == 'yes')
    reply_keyboard = [['Yes', 'No']]
    await update.message.reply_text(
        'Do you have a Certificate of Deposit (CD) with the bank?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return CD_ACCOUNT

async def cd_account(update: Update, context: CallbackContext) -> int:
    user_cd_account = update.message.text.lower() == 'yes'
    context.user_data['CD Account'] = int(user_cd_account)
    reply_keyboard = [['Yes', 'No']]
    await update.message.reply_text(
        'Do you use online banking facilities?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return ONLINE_BANKING

async def online_banking(update: Update, context: CallbackContext) -> int:
    user_online_banking = update.message.text.lower() == 'yes'
    reply_keyboard = [['Yes', 'No']]
    context.user_data['Online'] = int(user_online_banking)
    await update.message.reply_text(
        'Do you have a credit card issued by the bank? (Yes/No)',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return CREDIT_CARD

async def credit_card(update: Update, context: CallbackContext) -> int:
    user_credit_card = update.message.text.lower() == 'yes'
    context.user_data['CreditCard'] = int(user_credit_card)
    await update.message.reply_text('What is your average spending on credit cards per month? (In thousands of dollars)')
    return AVG_SPENDING

async def avg_spending(update: Update, context: CallbackContext) -> int:
    user_avg_spending = update.message.text
    context.user_data['CCAvg'] = float(user_avg_spending)

    with open('experiments/models/xgboost_original_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)  

    feature_names = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    user_data_values = [context.user_data.get(feature, 0) for feature in feature_names]
    user_data_for_prediction = pd.DataFrame([user_data_values], columns=feature_names)

    prediction = loaded_model.predict(user_data_for_prediction)

    await update.message.reply_text(f'Based on your information, we {("do not", "do")[prediction[0]]} recommend you to accept the personal loan offer.\n\nTo start again, send /start')

    return ConversationHandler.END

async def cancel(update: Update, _: CallbackContext) -> int:
    await update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END

def main():
    load_dotenv()
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    application = Application.builder().token(bot_token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age)],
            WORK_EXP: [MessageHandler(filters.TEXT & ~filters.COMMAND, work_experience)],
            INCOME: [MessageHandler(filters.TEXT & ~filters.COMMAND, income)],
            FAMILY_MEMBERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, family_members)],
            EDUCATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, education)],
            SECURITIES_ACCOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, securities)],
            CD_ACCOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, cd_account)],
            ONLINE_BANKING: [MessageHandler(filters.TEXT & ~filters.COMMAND, online_banking)],
            CREDIT_CARD: [MessageHandler(filters.TEXT & ~filters.COMMAND, credit_card)],
            AVG_SPENDING: [MessageHandler(filters.TEXT & ~filters.COMMAND, avg_spending)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
