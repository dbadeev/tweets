'''
Text cleaning funcs

- ignore case
- ignore punctuation
- remove stop words (optional)
- remove misspelling (optional)
'''

from spellchecker import SpellChecker
from nltk.corpus import stopwords
import re
import string
import contractions

# https://towardsdatascience.com/cleaning-text-data-with-python-b69b47b97b76


def remove_ticks_plus(text):
	return re.sub(r"\'\w+", '', text)


def remove_numbers(text):
	return re.sub(r'\w*\d+\w*', '', text)


def remove_hashtags(text):
	return re.sub("#\S+", " ", text)


def remove_mentions(text):
	return re.sub("@\S+", " ", text)


def remove_url(text):
	return re.sub("http*\S+", " ", text)


def remove_punctuations(text):
	return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)


def remove_extra_spaces(text):
	return re.sub(r'\s{2,}', " ", text)


def replace_emoticons(text):
	EMOTICONS = {
		r": )": "Happy face or smiley",
		r":)": "Happy face or smiley",
		r":-]": "Happy face or smiley",
		r":]": "Happy face or smiley",
		r":-3": "Happy face smiley",
		r":3": "Happy face smiley",
		r":->": "Happy face smiley",
		r":>": "Happy face smiley",
		r"8-)": "Happy face smiley",
		r":o)": "Happy face smiley",
		r":-}": "Happy face smiley",
		r":}": "Happy face smiley",
		r":-)": "Happy face smiley",
		r":c)": "Happy face smiley",
		r":^)": "Happy face smiley",
		r"=]": "Happy face smiley",
		r"=)": "Happy face smiley",
		r": D": "Laughing, big grin or laugh with glasses",
		r":D": "Laughing, big grin or laugh with glasses",
		r"8 D": "Laughing, big grin or laugh with glasses",
		r"8D": "Laughing, big grin or laugh with glasses",
		r"X D": "Laughing, big grin or laugh with glasses",
		r"XD": "Laughing, big grin or laugh with glasses",
		r"=D": "Laughing, big grin or laugh with glasses",
		r"=3": "Laughing, big grin or laugh with glasses",
		r"B^D": "Laughing, big grin or laugh with glasses",
		r":-))": "Very happy",
		r": (": "Frown, sad, andry or pouting",
		r":-(": "Frown, sad, andry or pouting",
		r":(": "Frown, sad, andry or pouting",
		r": c": "Frown, sad, andry or pouting",
		r":c": "Frown, sad, andry or pouting",
		r": <": "Frown, sad, andry or pouting",
		r":<": "Frown, sad, andry or pouting",
		r": [": "Frown, sad, andry or pouting",
		r":[": "Frown, sad, andry or pouting",
		r":-||": "Frown, sad, andry or pouting",
		r">:[": "Frown, sad, andry or pouting",
		r":{": "Frown, sad, andry or pouting",
		r":@": "Frown, sad, andry or pouting",
		r">:(": "Frown, sad, andry or pouting",
		r":' (": "Crying",
		r":'(": "Crying",
		r":' )": "Tears of happiness",
		r":')": "Tears of happiness",
		r"D ':": "Horror",
		r"D:<": "Disgust",
		r"D:": "Sadness",
		r"D8": "Great dismay",
		r"D;": "Great dismay",
		r"D=": "Great dismay",
		r"DX": "Great dismay",
		r": O": "Surprise",
		r":O": "Surprise",
		r": o": "Surprise",
		r":o": "Surprise",
		r":-0": "Shock",
		r"8 0": "Yawn",
		r">:O": "Yawn",
		r":-*": "Kiss",
		r":*": "Kiss",
		r":X": "Kiss",
		r"; )": "Wink or smirk",
		r";)": "Wink or smirk",
		r"*-)": "Wink or smirk",
		r"*)": "Wink or smirk",
		r"; ]": "Wink or smirk",
		r";]": "Wink or smirk",
		r";^)": "Wink or smirk",
		r": ,": "Wink or smirk",
		r";D": "Wink or smirk",
		r": P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r"X P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r"XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r": Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r"d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r"=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r": /": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r">:[(\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r":[(\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r"=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r"=[(\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r"=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
		r": |": "Straight face",
		r":|": "Straight face",
		r":$": "Embarrassed or blushing",
		r": x": "Sealed lips or wearing braces or tongue-tied",
		r":x": "Sealed lips or wearing braces or tongue-tied",
		r": #": "Sealed lips or wearing braces or tongue-tied",
		r":#": "Sealed lips or wearing braces or tongue-tied",
		r": &": "Sealed lips or wearing braces or tongue-tied",
		r":&": "Sealed lips or wearing braces or tongue-tied",
		r"O: )": "Angel, saint or innocent",
		r"O:)": "Angel, saint or innocent",
		r"0: 3": "Angel, saint or innocent",
		r"0:3": "Angel, saint or innocent",
		r"0: )": "Angel, saint or innocent",
		r"0:)": "Angel, saint or innocent",
		r": b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
		r"0;^)": "Angel, saint or innocent",
		r">: )": "Evil or devilish",
		r">:)": "Evil or devilish",
		r"}: )": "Evil or devilish",
		r"}:)": "Evil or devilish",
		r"3: )": "Evil or devilish",
		r"3:)": "Evil or devilish",
		r">;)": "Evil or devilish",
		r"|; )": "Cool",
		r"| O": "Bored",
		r": J": "Tongue-in-cheek",
		r"# )": "Party all night",
		r"% )": "Drunk or confused",
		r"%)": "Drunk or confused",
		r":-###..": "Being sick",
		r":###..": "Being sick",
		r"<: |": "Dump",
		r"(>_<)": "Troubled",
		r"(>_<)>": "Troubled",
		r"(';')": "Baby",
		r"(^^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
		r"(^_^;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
		r"(-_-;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
		r"(~_~;) (・.・;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
		r"(-_-)zzz": "Sleeping",
		r"(^_-)": "Wink",
		r"((+_+))": "Confused",
		r"(+o+)": "Confused",
		r"(o|o)": "Ultraman",
		r"^_^": "Joyful",
		r"(^_^)/": "Joyful",
		r"(^O^)／": "Joyful",
		r"(^o^)／": "Joyful",
		r"(__)": "Kowtow as a sign of respect, or dogeza for apology",
		r"_(._.)_": "Kowtow as a sign of respect, or dogeza for apology",
		r"<(_ _)>": "Kowtow as a sign of respect, or dogeza for apology",
		r"<m(__)m>": "Kowtow as a sign of respect, or dogeza for apology",
		r"m(__)m": "Kowtow as a sign of respect, or dogeza for apology",
		r"m(_ _)m": "Kowtow as a sign of respect, or dogeza for apology",
		r"('_')": "Sad or Crying",
		r"(/_;)": "Sad or Crying",
		r"(T_T) (;_;)": "Sad or Crying",
		r"(;_;": "Sad of Crying",
		r"(;_:)": "Sad or Crying",
		r"(;O;)": "Sad or Crying",
		r"(:_;)": "Sad or Crying",
		r"(ToT)": "Sad or Crying",
		r";_;": "Sad or Crying",
		r";-;": "Sad or Crying",
		r";n;": "Sad or Crying",
		r";;": "Sad or Crying",
		r"Q.Q": "Sad or Crying",
		r"T.T": "Sad or Crying",
		r"QQ": "Sad or Crying",
		r"Q_Q": "Sad or Crying",
		r"(-.-)": "Shame",
		r"(-_-)": "Shame",
		r"(一一)": "Shame",
		r"(；一_一)": "Shame",
		r"(=_=)": "Tired",
		r"(=^·^=)": "cat",
		r"(=^··^=)": "cat",
		r"=_^=   ": "cat",
		r"(..)": "Looking down",
		r"(._.)": "Looking down",
		r"^m^": "Giggling with hand covering mouth",
		r"(・・?": "Confusion",
		r"(?_?)": "Confusion",
		r">^_^<": "Normal Laugh",
		r"<^!^>": "Normal Laugh",
		r"^/^": "Normal Laugh",
		r"（*^_^*）": "Normal Laugh",
		r"(^<^) (^.^)": "Normal Laugh",
		r"(^^)": "Normal Laugh",
		r"(^.^)": "Normal Laugh",
		r"(^_^.)": "Normal Laugh",
		r"(^_^)": "Normal Laugh",
		r"(^J^)": "Normal Laugh",
		r"(*^.^*)": "Normal Laugh",
		r"(^—^）": "Normal Laugh",
		r"(#^.^#)": "Normal Laugh",
		r"（^—^）": "Waving",
		r"(;_;)/~~~": "Waving",
		r"(^.^)/~~~": "Waving",
		r"(-_-)/~~~ ($··)/~~~": "Waving",
		r"(T_T)/~~~": "Waving",
		r"(ToT)/~~~": "Waving",
		r"(*^0^*)": "Excited",
		r"(*_*)": "Amazed",
		r"(*_*;": "Amazed",
		r"(+_+) (@_@)": "Amazed",
		r"(*^^)v": "Laughing,Cheerful",
		r"(^_^)v": "Laughing,Cheerful",
		r"((d[-_-]b))": "Headphones,Listening to music",
		r'(-"-)': "Worried",
		r"(ーー;)": "Worried",
		r"(^0_0^)": "Eyeglasses",
		r"(＾ｖ＾)": "Happy",
		r"(＾ｕ＾)": "Happy",
		r"(^)o(^)": "Happy",
		r"(^O^)": "Happy",
		r"(^o^)": "Happy",
		r")^o^(": "Happy",
		r":O o_O": "Surprised",
		r"o_0": "Surprised",
		r"o.O": "Surpised",
		r"(o.o)": "Surprised",
		r"oO": "Surprised",
		r"(*￣m￣)": "Dissatisfied",
		r"(‘A`)": "Snubbed or Deflated"
	}

	for emot in EMOTICONS:
		text = text.replace(emot, " " + EMOTICONS[emot] + " ")
		# text = text.replace(emot, "_".join(EMOTICONS[emot].replace(",", "").replace(":", "").split()))
		# text = re.sub(r'(' + emot + r')', "_".join(EMOTICONS[emot].replace(",", "").replace(":", "").split()), text)
		# text = re.sub(u'(' + el + ')', " " + (" ".join(EMOTICONS[el].replace(
		# 	",", "").split())) + " ", text)
	return text


def misspellings(text: list) -> list:
	spell_corrector = SpellChecker()

	# initialize empty list to save correct spell words
	correct_words = []
	# extract spelling incorrect words by using unknown function of spellchecker
	misSpelled_words = spell_corrector.unknown(text)

	for each_word in text:
		if each_word in misSpelled_words:
			right_word = spell_corrector.correction(each_word)
			correct_words.append(right_word)
		else:
			correct_words.append(each_word)

	# 	# joining correct_words list into single string
	# 	correct_spelling = ' '.join(correct_words)
	# 	return correct_spelling
	return correct_words


def clean(text: str, remove_stop=False, misspelling=False) -> str:
	# Replace
	text = contractions.fix(text)
	text = replace_emoticons(text)
	# Remove
	text = remove_url(text)
	text = remove_hashtags(text)
	text = remove_ticks_plus(text)
	text = remove_mentions(text)
	text = remove_numbers(text)
	text = remove_punctuations(text)

	# Convert words to lower case and split them
	text = text.lower().split()

	# Remove stop words
	# if remove_stop:
	# 	stops = set(stopwords.words("english"))
	#
	# 	text = [w for w in text if not w in stops and len(w) >= 3]

	# Remove misspellings
	if misspelling:
		text = misspellings(text)

	text = " ".join(text)
	text = remove_extra_spaces(text)
	return text
