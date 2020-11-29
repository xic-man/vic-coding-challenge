from urllib.request import Request, urlopen  # For requesting raw html data
from datamuse import datamuse  # For generating rhyming words
from bs4 import BeautifulSoup  # For formatting and obtainging html data
import regex as re  # For obtaining specific string of text in raw data
import datetime  # For formatting and manipulating dates
from dateutil.relativedelta import relativedelta  # To calculate age of given person
import random  # To shuffle lists and select at random from a list
import wikipedia  # To obtain the gender of the person
from num2words import num2words  # To generate a worded version of their birth year for rhyming

import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR')

import model, sample, encoder  # importing the AI functions


def get_predicted_text(raw_text, model_name='345M', length=512, batch_size=1, temperature=0.9, top_k=40, top_p=0.9):
    """
    Run the sample_model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        dict2 = json.load(f)
        for key, value in hparams.items():
            hparams[key] = dict2[key]

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        out = sess.run(output, feed_dict={
          context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        return enc.decode(out[0])


def get_rhyming_words(word, words_to_return=10, words_to_generate=10, syllables=1, filter_noun=True):
    # Takes word to rhyme with, number of rhyming words to return, number of words to generate and number of syllables to filter by
    if words_to_generate < words_to_return:  # If words_to_generate is larger than words_to_return
        words_to_generate = words_to_return  # Set there to be no randomness in the output
    apiresult = datamuse.Datamuse().words(rel_rhy=str(word))  # Obtaining list of rhyming words
    while len(apiresult) < words_to_return:  # Adding words to the list so it fits the number of words to return, will result in duplicate words
        for i in range(0, len(apiresult)):
            words = datamuse.Datamuse().words(rel_rhy=str(apiresult[i]["word"]))
            if words != []:
                for k in words:
                    apiresult.append(k)
                    if len(apiresult) >= words_to_return:
                        break
        if apiresult == []:
            return apiresult
            break
    i = 0
    while i <= syllables:
        j = 0
        while j < len(apiresult) and len(apiresult) > words_to_generate:  # Filtering out words with less than the given number of syllables
            if apiresult[j]["numSyllables"] == i:
                apiresult.pop(j)
            else:
                j += 1
        i += 1
    if len(apiresult) > words_to_generate:
        apiresult = apiresult[:words_to_generate]
    rhyming_words = []
    for i in apiresult:
        rhyming_words.append(i['word'])  # Appending only the words to the output list
    return random.sample(rhyming_words, words_to_return)


def get_date(type, raw_data):  # Takes type (born or died) to figure out significant dates, along with raw data
    # Different formatted date types are searched for below, ordered inverted based on importance (least important first)
    # This is done so the date_formatted variable is always defined as accurately as it can be, as it is redefined if a better format is found
    # This way of doing it is slightly inefficient, but works quite reliably
    date_format = "%Y-%m-%d"  # the format that all dates are put into
    date_formatted = "unknown"  # Used if no date can be found
    # Obtaining the birthdate using regex (should work with almost all birthday prefix formats)

    if type == 'died':  # Set death date to today. Keeps it that way if a death date is not found in the article. Used to calucate current age.
        date_formatted = datetime.datetime.strptime(datetime.datetime.now().date().strftime(date_format), date_format)

    # just int year type (e.g: 1970)
    if type == "born":
        date_raw = re.search(r'(?<=(Date of birth|Born)[^,]*)[0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    elif type == "died":
        date_raw = re.search(r'(?<=(Date of death|Died)[^,]*)[0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    if date_raw is not None:
        # print("just int year type")
        if len(date_raw.group(0).replace(" ", "")) <= 4:
            # Adding a zero to the start of the year if it is less than 3 digits long to fit format
            year_prefix = (4 - len(date_raw.group(0).split()[0])) * "0"  # e.g 197 -> 0197 to fit in 01-01-0197
            date_formatted = str(year_prefix) + str(date_raw.group(0).replace(" ", "")) + "-01-01"
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)

    # no day, word month, int year type (e.g: January 1970)
    if type == "born":
        date_raw = re.search(r'(?<=(Date of birth|Born)[^,]*)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) [0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    elif type == "died":
        date_raw = re.search(r'(?<=(Date of death|Died)[^,]*)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) [0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    if date_raw is not None:
        # print("no day, word month, int year type")
        # Adding a zero to the start of the year if it is less than 3 digits long to fit format
        year_prefix = (4 - len(date_raw.group(0).split()[1])) * "0"  # e.g 197 -> 0197 to fit in 01-01-0197
        i = datetime.datetime.strptime(date_raw.group(0).split()[0], "%B").month
        if i < 10:  # Adding a zero to the month if it is less than 10 (e.g 01-1-1970 -> 01-01-1970)
            date_formatted = str(year_prefix) + str(date_raw.group(0).split()[1]) + "-0" + str(i) + "-01"
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)  # ValueError: time data '4-08-01' does not match format '%Y-%m-%d'
        else:
            date_formatted = str(year_prefix) + str(date_raw.group(0).split()[1]) + "-" + str(i) + "-01"
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)

    # word month, int day and int year type (January 1, 1970)
    if type == "born":
        date_raw = re.search(r'(?<=(Date of birth|Born)[^,]*)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) [0-9]{1,2}\, [0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    elif type == "died":
        date_raw = re.search(r'(?<=(Date of death|Died)[^,]*)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) [0-9]{1,2}\, [0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    if date_raw is not None:
        # print("word month, int day and int year type")
        date_raw = re.sub(',', '', date_raw.group(0), flags=re.IGNORECASE)
        # Adding a zero to the start of the year if it is less than 3 digits long to fit format
        year_prefix = (4 - len(date_raw.split()[2])) * "0"  # e.g 197 -> 0197 to fit in 01-01-0197
        i = datetime.datetime.strptime(date_raw.split()[0], "%B").month
        if i < 10: # Adding a zero to the month if it is less than 10 (e.g 01-1-1970 -> 01-01-1970)
            date_formatted = str(year_prefix) + str(date_raw.split()[2]) + "-0" + str(i) + "-" + str(date_raw.split()[1])
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)  # ValueError: time data '4-08-01' does not match format '%Y-%m-%d'
        else:
            date_formatted = str(year_prefix) + str(date_raw.split()[2]) + "-" + str(i) + "-" + str(date_raw.split()[1])
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)

    # int day and year, word month type (e.g: 1 january 1970)
    if type == "born":
        date_raw = re.search(r'(?<=(Date of birth|Born)[^,]*)[0-9]{1,2} \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) [0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    elif type == "died":
        date_raw = re.search(r'(?<=(Date of death|Died)[^,]*)[0-9]{1,2} \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) [0-9]{1,4}', raw_data, flags=re.IGNORECASE)
    if date_raw is not None:
        # print("int day and year, word month type")
        # Adding a zero to the start of the year if it is less than 3 digits long to fit format
        year_prefix = (4 - len(date_raw.group(0).split()[2])) * "0"  # e.g 197 -> 0197 to fit in 01-01-0197
        # Converting worded month to digit month, using walrus operator to save space
        i = datetime.datetime.strptime(date_raw.group(0).split()[1], "%B").month
        if i < 10:  # Adding a zero to the month if it is less than 10 (e.g 01-1-1970 -> 01-01-1970)
            # Compiling all data and putting it into format, then properly formatting it just in case
            date_formatted = str(year_prefix) + str(date_raw.group(0).split()[2]) + "-0" + str(i) + "-" + str(date_raw.group(0).split()[0])
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)
        else:
            date_formatted = str(year_prefix) + str(date_raw.group(0).split()[2]) + "-" + str(i) + "-" + str(date_raw.group(0).split()[0])
            date_formatted = datetime.datetime.strptime(date_formatted, date_format)

    # Normal birthday type (e.g: 01-01-1970)
    if type == "born":
        date_raw = re.search(r'(?<=(Date of birth|Born)[^,]*)[0-9]{1,4}-[0-9]{1,2}-[0-9]{1,2}(?= \))', raw_data, flags=re.IGNORECASE)
    elif type == "died":
        date_raw = re.search(r'(?<=(Date of death|Died)[^,]*)[0-9]{1,4}-[0-9]{1,2}-[0-9]{1,2}(?= \))', raw_data, flags=re.IGNORECASE)
    if date_raw is not None:
        # print("Normal birthday type")
        date_formatted = datetime.datetime.strptime(date_raw.group(0), date_format)
    try:
        date_found = date_formatted.date()
    except AttributeError:
        date_found = "unknown"
    return date_found


def get_data(full_name):
    # Get first and last name of the person and correct possible formatting errors
    try:
        full_name = wikipedia.search(full_name)[0]
        try:
            first_name = re.sub(r'\,|\"|\'|\(|\)|\{|\}|\[|\]|\||\\|\/|\?|\!|\@|\#|\$|\%|\^|\&|\*|\_|\+|\=|\:|\;|\<|\>|\,|\.', '', full_name.split()[0].replace(" ", "").replace(",", ""), flags=re.IGNORECASE).title()
            last_name = re.sub(r'\,|\"|\'|\(|\)|\{|\}|\[|\]|\||\\|\/|\?|\!|\@|\#|\$|\%|\^|\&|\*|\_|\+|\=|\:|\;|\<|\>|\,|\.', '', re.sub(r'.*? ', '', full_name, 1), flags=re.IGNORECASE).title()
        except AttributeError:
            print(f"Please type in a more specific name (currently {full_name})\n")
            return
        url = "https://en.wikipedia.org/wiki/" + full_name.replace(" ", "_")
    except IndexError:
        print(f"Check spelling of {full_name} and try again\n")
        return
    except TimeoutError:
        print('Connection to wikipedia timed out, please check your internet connection\n')
        exit()
    except wikipedia.exceptions.WikipediaException:
        print("Name cannot be empty, please type in a name\n")
        return

    # Obtaining gender of person using name variable
    gender = "unknown"
    if full_name != "":
        try:
            page = wikipedia.page(full_name, auto_suggest=False).content
        except:
            print('Ambiguous name submitted, please be more specific\n')
            return
        try:
            pronouns = re.search(r'(?<![a-z])she(?![[a-z],\'])|(?<![a-z])her(?![[a-z],\'])|(?<![a-z])he(?![[a-z],\'])|(?<![a-z])his(?![[a-z],\'])', page, flags=re.IGNORECASE).group(0).lower()
        except AttributeError:
            print(f"Please type in a more specific name (currently {full_name})\n")
            return
        if pronouns == "she" or pronouns == "her":
            gender = "female"
        elif pronouns == "he" or pronouns == "his":
            gender = "male"
    career = None
    # Checking if the name is in the format (first_name last_name (career)) and obtaining career from it
    career_bracket = re.search(r'(?<=\().*?(?=\))', full_name, flags=re.IGNORECASE)
    if career_bracket is not None:  # e.g John Smith (politician)
        career = career_bracket.group(0)

    # Fetching the raw HTML content
    try:
        html_content = str(urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})).read())
    except UnicodeEncodeError:  # FIX
        print(urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})))
        print(type(urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}))))
        html_content = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})).encode('utf-8').strip()
    # Parsing the html content
    soup = BeautifulSoup(html_content, "html.parser")
    # Obtaining the html of the infobox of the person from a table with the class infobox
    infobox_table = soup.find("table", attrs={"class": "infobox biography vcard"})
    # Sometimes the html attribute of the infobox is "infobox vcard" or "infobox vcard plainlist" (only for musicians)
    if infobox_table is None:
        infobox_table = soup.find("table", attrs={"class": "infobox vcard plainlist"})
        if infobox_table is not None:
            career = "musician"

    if infobox_table is None:
        infobox_table = soup.find("table", attrs={"class": "infobox vcard"})

    try:
        infobox_table_data = infobox_table.tbody.find_all("tr")  # Only obtaining the data contained in rows in the infobox
    except:
        print(f"Note: Data could not be obtained, please check spelling of {full_name} and try again\n")
        return
    table_data_html = str(infobox_table_data)

    # Getting rid of html tags and characters using regex
    table_data_cleaned = re.sub('<[^<]+?>', ' ', table_data_html, flags=re.IGNORECASE)
    # print(table_data_cleaned)
    # Replacing any number of spaces in a row with just one (from "    " or "  " to " ")
    table_data_cleaned = ' '.join(table_data_cleaned.split())

    # obtaining the birth and death date using get_date function
    birth_date = get_date("born", table_data_cleaned)
    death_date = get_date("died", table_data_cleaned)

    if death_date == datetime.datetime.strptime(datetime.datetime.now().date().strftime('%Y-%m-%d'), '%Y-%m-%d').date():
        alive_or_dead = "alive"
        age = str(relativedelta(death_date, birth_date).years).replace(" ", "")  # Calculating age using relativedelta
    elif death_date == "unknown":
        age = "unknown"
    else:
        alive_or_dead = "dead"
        age = str(relativedelta(death_date, birth_date).years).replace(" ", "")  # Calculating age using relativedelta

    if int(age) > 122:  # In case the death date is not obtained correctly, and is set to today
        age = "unknown"

    # Searching for the keywords of specific careers to find their career
    science = re.search(r'scientific career', table_data_cleaned, flags=re.IGNORECASE)
    politics = re.search(r'(in|assumed) office', table_data_cleaned, flags=re.IGNORECASE)
    musician = re.search(r'musician|instrument|musical', table_data_cleaned, flags=re.IGNORECASE)
    actor = re.search(r'actor', table_data_cleaned, flags=re.IGNORECASE)
    comedian = re.search(r'comedian|comedy', table_data_cleaned, flags=re.IGNORECASE)
    journalist = re.search(r'journalist|journal|radio', table_data_cleaned, flags=re.IGNORECASE)
    sports = re.search(r'sport|coach|pro(?![a-z])|(?<![a-z])team(?![a-z])', table_data_cleaned, flags=re.IGNORECASE)
    author = re.search(r'author|writer|poet|novelist|playwright', table_data_cleaned, flags=re.IGNORECASE)
    military = re.search(r'rank|commands|years of service|allegiance', table_data_cleaned, flags=re.IGNORECASE)
    artist = re.search(r'movement|notable work', table_data_cleaned, flags=re.IGNORECASE)
    subscience = None  # Default value
    # Assigning career
    if career is None:  # Just in case career is defined through the persons name
        if science is not None:
            career = "scientist"
            # Obtaining a more specific scientific career
            subscience = re.search(r'(?<=Fields ).+?(?=,)', table_data_cleaned, flags=re.IGNORECASE)
            if subscience is not None:
                subscience = str(subscience.group(0)).split(" ")[0].lower()  # 1st result of regex search and 1st word in result
        elif politics is not None:
            career = "politician"
        elif sports is not None:
            career = "sports person"
        elif actor is not None:
            career = "actor"
        elif musician is not None:
            career = "musician"
        elif comedian is not None:
            career = "comedian"
        elif journalist is not None:
            career = "journalist"
        elif author is not None:
            career = "author"
        elif military is not None:
            career = "military personnel"
        elif artist is not None:
            career = "artist"
        else:
            career = "unknown"

    return str(birth_date), age, alive_or_dead, career, gender, subscience, first_name, last_name, url


def generate_poem(all_data, poem_settings):
    poem_order = poem_settings["Poem order"]
    ai_fill = poem_settings["AI fill"]
    set_order = poem_settings["Set poem order"]
    words_to_generate = poem_settings["Words to generate"]
    number_of_syllables = poem_settings["Number of syllables"]

    full_name = all_data[0]
    first_name = all_data[1]
    last_name = all_data[2]
    age = all_data[3]
    birth_year = all_data[4]
    alive_or_dead = all_data[5]
    career = all_data[6]
    gender = all_data[7]
    subscience = all_data[8]

    # Rhyming code
    lastname_initial = last_name[0]
    lastname_initial_rhyme = get_rhyming_words(lastname_initial, 5, words_to_generate, number_of_syllables)

    if birth_year != "year":
        # Converting integer year to words using num2words
        if len(str(birth_year)) == 4:  # Testing if the year has 4 digits
            # splitting year into two numbers and getting words for these numbers, then combining them
            full_year_name = num2words(str(birth_year)[0:2]) + " " + num2words(str(birth_year)[2:4])
            if str(birth_year)[1] == "0" and str(birth_year)[2] == "0":  # Checking for a year with format X00X where X != 0
                full_year_name = num2words(birth_year)
            if str(birth_year)[1] != "0" and str(birth_year)[2] == "0":  # Checking for year with format XY0X in which Y != 0
                # Converting from XY0X format to a worded format (e.g 1902 -> nineteen o'two)
                full_year_name = num2words(str(birth_year)[0:2]) + " o'" + num2words(str(birth_year)[3])
        else:  # If the year is less than 4 digits long, word is the entire number
            full_year_name = num2words(birth_year)

        # Obtaining the last word of birth_year, used to rhyme with in poem
        year_word_to_rhy = full_year_name.split("-")[len(full_year_name.split("-")) - 1]  # Obtaining the last word of the string
        if year_word_to_rhy == str(full_year_name):
            year_word_to_rhy = full_year_name.split()[len(full_year_name.split()) - 1]  # Obtaining the last word of the string

        year_rhymes = get_rhyming_words(year_word_to_rhy, 1, words_to_generate, number_of_syllables)  # Obtaining rhymes
    else:
        year_rhymes = ["unknown"]
        year_word_to_rhy = "unknown"

    if career == "sports person":  # Rhyme with last word of profession instead of both words
        career_rhymes = get_rhyming_words("person", 5, words_to_generate, number_of_syllables)
    elif career == "military personnel":  # Rhyme with last word of profession instead of both words
        career_rhymes = get_rhyming_words("personnel", 5, words_to_generate, number_of_syllables)
    elif career == "scientist":  # Had to manually set the rhyming words of scientist, as datamuse returns []
        career_rhymes = ["enlist", "dentist", "insist", "fist",  "rightist", "slightest",  "sweetist", "subsist", "catalyst"]
    else:
        career_rhymes = get_rhyming_words(career, 5, words_to_generate, number_of_syllables)

    amount_desc = ["very", "vastly", "hugely", "perfectly", "largely"]
    positive_desc = ["good", "amazing", "nice", "brilliant", "cool", "fantastic", "awesome", "sensational", "legendary", "epic"]
    chosen_positive_desc = random.choice(positive_desc)

    # Changing list based on whether the person is alive, dead, male or female
    auxiliary_verbs = ["they", "their", "was", "were", "lived to", "included"]  # default list if current state is unknown
    if alive_or_dead == "alive":
        if gender == "male":
            auxiliary_verbs = ["he", "his", "is", "are", "is currently", "includes"]
        if gender == "female":
            auxiliary_verbs = ["she", "her", "is", "are", "is currently", "includes"]

    if alive_or_dead == "dead":
        if gender == "male":
            auxiliary_verbs = ["he", "his", "was", "were", "lived to", "included"]
        if gender == "female":
            auxiliary_verbs = ["she", "her", "was", "were", "lived to", "included"]

    # Optional subscience rhyme, extension to line1
    line1_extension = ""
    if subscience is not None:
        line1_extension = ", " + random.choice(amount_desc) + " " + random.choice(positive_desc) + " at " + subscience + " as well"
    # Optional name rhyme, extension to line2
    line2_extension = ""
    name_rhyme = get_rhyming_words(first_name, 5, words_to_generate, number_of_syllables)  # Attempting rhyme with first name
    if name_rhyme == []:
        if len(full_name.split()) == 2:
            name_rhyme = get_rhyming_words(full_name.split()[1], 5, words_to_generate, number_of_syllables)  # Attempting rhyme with last name
        if name_rhyme == [] and len(full_name.split()) >= 3:
            name_rhyme = get_rhyming_words(full_name.split()[2], 5, words_to_generate, number_of_syllables)  # Attempting rhyme with middle name
    if name_rhyme != []:
        line2_extension = "Owning vast amounts of " + name_rhyme[0] + ", " + name_rhyme[1] + ", and " + name_rhyme[2] + ", " + "\n"

    full_name = re.sub(r'\ *\(.*\)', '', full_name, flags=re.IGNORECASE)
    # Poem compilation
    # Pattern A E.G: John                  Smith                         is             a               very                           good                 politician
    line1 = first_name.title() + " " + last_name.title() + " " + auxiliary_verbs[2] + " a " + random.choice(amount_desc) + " " + chosen_positive_desc + " " + career + line1_extension
    # Pattern A E.G:              He              is curently               50         years old and            is                              vastly                          intuition
    line2 = line2_extension + auxiliary_verbs[0].title() + " " + auxiliary_verbs[4] + " " + str(age) + " years old and " + auxiliary_verbs[2] + " " + random.choice(amount_desc) + " " + career_rhymes[0]
    # Pattern A E.G:              his                                     goodness                is             admired by many,          his                 dispositiionness as well
    line3 = auxiliary_verbs[1].title() + " " + chosen_positive_desc + "ness " + auxiliary_verbs[2] + " admired by many, " + auxiliary_verbs[1] + " " + career_rhymes[1] + "ness as well"
    # Pattern A E.G: Some sasy that John      would sometimes go and       commission        ,        condition        , and      juxtaposition
    line4 = "Some say that " + first_name + " would sometimes go and get a " + career_rhymes[2] + ", " + career_rhymes[3] + ", and " + career_rhymes[4]
    # Pattern B E.G: John                     S.                  may have owned a               mess
    line5 = first_name.title() + " " + lastname_initial + "." + " may have owned a " + lastname_initial_rhyme[0]
    # Pattern B E.G:      His              large collection of things           includes         a             bless                , a             finesse              , a                ness              , and a           chest
    line6 = auxiliary_verbs[1].title() + " large collection of things " + auxiliary_verbs[5] + " a " + lastname_initial_rhyme[1] + ", a " + lastname_initial_rhyme[2] + ", a " + lastname_initial_rhyme[3] + ", and a " + lastname_initial_rhyme[4]
    # Pattern C E.G: John         was born in     nineteen seventy
    line7 = last_name.title() + " was born in " + full_year_name
    # Pattern C E.G: His                   birth was            great            , not mentioning the      seven trees
    line8 = auxiliary_verbs[1].title() + " birth was " + chosen_positive_desc + ", not mentioning the " + year_rhymes[0]

    list_of_lines = [line1, line2, line3, line4, line5, line6, line7, line8]
    a_lines = [line1, line2, line3, line4]
    b_lines = [line5, line6]
    c_lines = [line7, line8]
    raw_poem = ""
    for i in list_of_lines:
        raw_poem += i + "\n"

    no_of_words = 0
    for i in list_of_lines:  # Calculating the number of words in all lines of the poem
        no_of_words += int(len(i.split()))

    if poem_order == "random":  # Shuffling list and printing the shuffle
        random.shuffle(list_of_lines)
        for i in list_of_lines:
            print(i)
        return

    for i in set_order:  # Iterating through set order and printing based on characters
        if i == "a":
            print(a_lines[0])
            a_lines.pop(0)
        if i == "b":
            print(b_lines[0])
            b_lines.pop(0)
        if i == "c":
            print(c_lines[0])
            c_lines.pop(0)
    if ai_fill:
        AI_output = ""
        testing_no_of_words = no_of_words
        AI_text = get_predicted_text(raw_poem, model_name='345M', length=512, batch_size=1, temperature=0.9, top_k=40, top_p=0.9)
        AI_text = re.sub(r'\<\|endoftext\|\>.*', '', AI_text, flags=re.IGNORECASE)
        AI_text = re.sub(r'\.(\n| )|\n', '\n', AI_text, flags=re.IGNORECASE)
        for i in AI_text.split('\n'):
            if i in AI_output:
                continue
            for j in i.split():
                testing_no_of_words += 1
            if testing_no_of_words >= 200:
                break
            no_of_words = testing_no_of_words
            AI_output += i
        print(AI_output)
    print(f"\nAbove poem is {no_of_words} words long")
