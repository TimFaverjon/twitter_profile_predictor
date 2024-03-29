# Twitter Profile Predictor

WELCOME to Twitter profile predictor, a quick and easy way to estimate professions, statuses, topics and other features of Twitter users from their description in an auditable way. And suitable for FRENCH !

For any question contact : tim.faverjon@sciencespo.fr.

## Description

This package is designed to predict the characteristics of a Twitter profile based on bios.

Predicted characteristics :
- language (language model detection)
- professions
- declared statuses
- gender
- age
- topics

Detailed documentation can be found [here](https://timfaverjon.github.io/twitter_profile_predictor/bios_analyzer.html).

### Methodology

Language is identified using language models.

All the other characteristics are estimated by identifying keywords and bi-words in the bios. The keywords have been choosen by human annotation among the most used words and bi-words in a sample of French Twitter users (see more detail in the documentation section). 

## Installation

To install this package run the command:

```shell
pip install --upgrade twitter-profile-predictor
```

The package is uploaded on [PyPI](https://pypi.org/project/twitter-profile-predictor/).

### WARNING

This package use [cld2-cffi](https://pypi.org/project/cld2-cffi/), you may have problem installing `cld2-cffi` while installing `twitter-profile-predictor`. If it's the case please consider installing `cld2-cffi` in your environment before and then install `twitter-profile-predictor`.

## Example of use

```python
import twitter_profile_predictor as tpp

# Load the module
extractor = tpp.bios_analyzer("je suis president du monde et journaliste chez sciencespo")

# Tokenization
print('tokenization')
extractor.full_tokenize()
# The tokens are saved in the attribute full_tokens
print(extractor.full_tokens)

# Profession identification
print("profession extraction")
# The professions are saved in the attribute professions
print(extractor.get_professions())
```

You can find a [tutorial with examples](test/example_code.ipynb).

# Documentation

 In this section we discuss the scientific methodology. Detailed and complete documentation of the functions and classes can be found [here](https://timfaverjon.github.io/twitter_profile_predictor/bios_analyzer.html).

Our methodology is based on the analyisis of X (ex-Twitter) bios (also called description). Around 48% of all X users have a bios accessible on their profile[@culottaPredictingTwitterUser2016]. Our method have been developped on a specific subset of 30.000 users followers of french members of parliament. For this reason we recommended it for french datasets

## Language

The first analysis we made was to estimate the language of each user.
Giving the difficulty of the task on very small texts we use 3 different
available python language detector :
[`langid.classify`](https://github.com/saffsd/langid.py/blob/master/README.rst),
[`langdetect.detect`](https://pypi.org/project/langdetect/) and
[`cld3.get_language`](https://github.com/google/cld3), and we then get
the right language by majority vote or return `False` if no language
matches with another.

### Language in our dataset

| **Language**      | **Number of users** |
|-------------------|---------------------|
| French            | 17956               |
| English           | 2449                |
| Spanish           | 132                 |
| Italian           | 71                  |
| German            | 62                  |
| Arabic            | 43                  |
| Portuguese        | 28                  |
| Catalan           | 24                  |
| Others            | 181                 |
| *Unrecognized*    | 1141                |

Table: Main languages used in the description of users on Twitter

We can see in the table above the main languages used by the users. The great majority (92.4%) of the descriptions are in French or English. Some users are non-French speakers. These users are here because they participate actively in the French political debate, following French MPs and being connected to the rest of the digital public sphere.

### Code 

You can call language using :

```python
import twitter_profile_predictor as tpp

bios = "example of twitter bios"

# direct function
tpp.lang(bios, mainfrench = False)

# bios analyser module
extractor = tpp.bios_analyzer(tpp)
extractor.get_lang()
extractor.language
```

The language returned as string with ISO 639-1 Language Codes (Two-letter codes):
- English: "en"
- French: "fr"
- Spanish: "es"
- German: "de"
- Chinese: "zh"
- Japanese: "ja"
- Russian: "ru"
- Arabic: "ar"
- Hindi: "hi"
- Portuguese: "pt"
- ecc...

## Profession

Estimating Twitter users' occupation from profile information is a common task in computational social sciences, particularly in attempts to assess socioeconomic status (SES) [@ghazouaniAssessingSocioeconomicStatus2019]. SES is considered a set of material, cultural, and social capital, clearly related to the profession.

The easiest way to estimate users' occupation is to use the auto-declared occupation in the user's profile description. According to [@preotiuc-pietroAnalysisUserOccupational2015], 20% of active users declare their occupational status in their description.

A common technique is to automatically search for professional occupations in the description and link them to standardized classifications such as the *nomenclature des Professions et Catégories Socioprofessionnelles* (CSP2020) for France, the *Standard Occupation Classification* (SOC2020) [@SOC2020Office] for the UK or USA, or the *International Standard Classification of Occupations* (ISCO-08) [@heMethodEstimatingIndividual2023; @sloanWhoTweetsDeriving2015].

There are four main difficulties in this task for assessing SES [@sloanWhoTweetsDeriving2015]:

1. Distinguishing main occupations among hobbies or secondary occupations[^1].
2. Accessing representativity on Twitter among different classes.
3. Ensuring that the use of given professional keywords is actually auto-declarative.
4. Wrong and/or biased auto-declared professions.

We tackle these problems by replacing automatic detection of profession with human-annotated professions, focusing mainly on the most common declared occupations.

Here's our method:

1. Tokenize the description, removing punctuation and stop words (in English and French).
2. Extract the most common words and bi-words.
3. Recognize words (or bi-words) referring to auto-declaration of an occupation (e.g., journalist, professor, city councilor...).
4. Link the declared occupation with standard classifications (CSP2020, SOC2020, ISCO-08).
5. Check (by human verification) if the use of the keyword in bios actually refers to auto-declaration of professions and remove keywords with accuracy under 80%.
6. Highlight the classes and sub-classes relevant for our analysis.

This technique not only allows us to obtain higher accuracy due to human verification but mainly allows us to identify more users by adapting the method to our specific dataset.

Here's how we tackle the challenges cited before:

1. In this work, we don't try to predict SES but only to estimate types of occupation to link them with online sharing behaviors. We allow users to label multiple professions for users who declare more than one.
2. Representativity is not a problem because we do not aim to make real-world scale statistics but to understand how the profession of users influences the recommendation algorithm specifically on Twitter.
3. Statistical studies present 30% of misclassification by automatic keyword detection. We rely on human verification and annotation, considering status words apart from professions, and checking the use of keywords with accuracy over 80%[^2].
4. We acknowledge that what we measure are only "declared professions." We concentrate on the most used words and bi-words, ensuring statistical significance of the result and reducing the risk of misclassification.

Because our dataset is 80% French, we choose to analyze the professions collected using the French standard classification CSP2020. Table 1 shows the most important categories identified in our dataset. We notice an over-representation of highly educated positions (Cadres et professions intellectuelles supérieures) from information, politics, and business, which has already been observed and is mainly due to the fact that professionals from those categories have a higher probability of stating their professional titles on their description.

| Professional category                             | Classification CSP2020 (French)                          | Users | Example of professions        |
|----------------------------------------------------|----------------------------------------------------------|-------|--------------------------------|
| Information, arts, and entertainment professions   | Professions de l’information, de l’art et des spectacles (3500) | 2315  | Journalist, producer...         |
| Business, IT, and administration professionals     | Cadres et professions intermédiaires administratives, commerciales et techniques des entreprises (3700, 3800, 4600) | 2160  | Developer, CEO...               |
| Elected officers and political representatives     | Cadres administratifs et techniques de la fonction publique (3300) | 1192  | Mayor, minister...              |
| Professors and higher scientific professions        | Professeurs et professions scientifiques supérieures (3400) | 636   | Professor, historian...         |
| Other                                              | Recognized professions in other CSP categories            | 530   | Nurse, architect...             |

[^1]: For example, "Engineer and gardener on the weekend." Is that an engineer or a hunter? "Actor, producer, film director, and writer." What is the main profession?

[^2]: For instance, we notice that plural occupation keywords (e.g., "parliamentarians") most often don't refer to auto-declaration of profession.

### Code 

You can get the list of professions and keywords by calling :

```python
tbb.get_pro_kewords()
```

## Declared Status

In addition to occupations, Twitter users often use words indicating their status. We used the same method as for occupations, examining the most frequent words and bi-words, and identified keywords indicating a status associated with the user. We found at least one status for 31% of the users. Here are the different statuses we have identified.

**Professional Status:**

These keywords provide information about users' occupations but indicate a status or position rather than a sector or profession. Here are the different identified statuses: advisor, director, president, founder, enthusiast, leader, assistant, manager, student, executive, delegate, alumni, candidate, vice president, professional, specialist, independent, elected, PhD, expert, important member, secretary-general, ambassador, collaborator, administrator, coordinator, doctor, board member, deputy director, apprentice, master, technician, worker. Identifying these statuses allows us to avoid misclassification problems related to professions.

**Type of Actor:**

On Twitter, many accounts do not belong to individual users but to organizations. To distinguish one from the other, we identified a series of keywords characterizing the type of actor owning the account. List: association, official account, personal account, university, media, committee, federation, agency, startup, company, union, foundation, city hall, think tank. Thanks to this status, we can distinguish official and personal accounts. These types of keywords are less precise than others for determining account ownership, but we judge them reasonable to use.

**Affiliation to a Group:**

These statuses indicate a declaration of affiliation to a political or cultural group. The statuses do not specify which affiliation it is, only if it exists. List: Activist, member, citizen, supporter, fan, volunteer.

**Degree and University:**

In France, certain universities and degrees can impact identity, indicating membership in a social group. Among the most used words, we recognized some of these universities. List: Sciences Po, Sorbonne, Mines.

### Code 

You can get the list of statuses and keywords by calling :

```python
tbb.get_status_kewords()
```

## Age

The age of users is delicate because we do not have verification methods to assess our approach. However, here we are not attempting to estimate the age of all users but only that of a few users. So, we choose to rely on some keywords that allow us to associate users with a notion of youth or old age. Here are the identified keywords.

*Old*: retraité, retraitée, senior, vieux, ex, exjournaliste, exprésident.

*Young*: étudiant, student, etudiant, étudiante, etudiante, jeune, junior.

We obtain an age for 2.2% of the users.

### Code 

You can get the list of age keywords by calling :

```python
tbb.get_age_kewords()
```

## Gender

The last demographic information we try to estimate is gender. In our dataset, very few users declare their pronouns, so it is not feasible to use this technique to estimate the gender of users. However, French (the predominant language in our dataset) is a gendered language, allowing us to identify the gender of the author (e.g., "je suis infirmier" indicates a male subject, while "je suis infirmière" indicates a female subject).

We then face two problems: (1) there is no model in French that can determine with certainty whether a word is feminine and whether it refers to the subject of the sentence; (2) if we identify feminine subjects, how can we ensure that they are indeed self-declarative and refer to a person of feminine gender rather than another feminine entity (company, association, etc.). To address these issues, we have chosen to use the previously identified keywords to determine the status and occupation. Indeed, we have already taken care to determine that these keywords are self-referential, and in the case of professional keywords, we know they refer to an individual rather than an entity.

So we review the self-declarative keywords, and when possible, we note if these keywords are specifically masculine or specifically feminine. Here are some examples of identified keywords.

*Masculine*: président, conseiller, directeur, chef, passionné, consultant, citoyen, délégué...

*Feminine*: conseillère, directrice, ingénieure, présidente, experte, citoyenne, passionnée...

We notice that this method introduces a bias, as we can identify the gender only of users declaring an occupation or status. Additionally, we observe that in French, the masculine is considered neutral, so for certain professions, we can only identify the feminine (e.g., "docteure" is feminine, but "docteur" can be used for both male and female). Therefore, we have a slight selection bias towards women. Finally, given the unbalanced population, we expect to identify more men than women.

However, because of the big size of our sample, we judge that we can still identify gender effects in recommender systems using this method.

We obtain a gender for 20.5% of the users (14.5% men, 6% women).

### Code 

You can get the list of gender keywords by calling :

```python
tbb.get_gender_kewords()
```

## Topics

As we are interested in the recommendation mechanism, it is sometimes valuable to know the interests of users in our dataset (e.g., politics, digital, communication, etc.), as these interests could strongly influence the consumed content. Using the same technique as for professions, we have identified and classified keywords referring to specific topics in users' bios. These words are not intended to indicate the discipline, occupation, or profile of the user but simply an affinity with a topic.

Here are the identified topics: politics, digital, communication, culture, public sector, innovation, law, education, tech, sciences, health, business, research, marketing, development, music, management, economy, environment/biodiversity, ecology, security, entrepreneurship, data, history, finance, press, energy, justice, agriculture, recruitment, social media, strategy, climate, real estate, construction, tourism, sustainable development, literature, mobility, diplomacy, agri-food, design, philosophy, sociology, engineering.

### Code 

You can get the list of topics and keywords by calling :

```python
tbb.get_topic_kewords()
```

## Bibliography 

[@culottaPredictingTwitterUser2016]: Culotta, A., Ravi, N. K., & Cutler, J. (2016). Predicting Twitter User Demographics Using Distant Supervision from Website Traffic Data. Journal of Artificial Intelligence Research, 55, 389-408. doi:10.1613/jair.4935.

[@ghazouaniAssessingSocioeconomicStatus2019]: Ghazouani, D., Lancieri, L., Ounelli, H., & Jebari, C. (2019). Assessing Socioeconomic Status of Twitter Users: A Survey. Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019), 388-398. doi:10.26615/978-954-452-056-4_046.

[@preotiuc-pietroAnalysisUserOccupational2015]: Preoţiuc-Pietro, D., Lampos, V., & Aletras, N. (2015). An Analysis of the User Occupational Class through Twitter Content. Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 1754-1764. doi:10.3115/v1/P15-1169.

[@SOC2020Office]: SOC 2020 - Office for National Statistics.

[@sloanWhoTweetsDeriving2015]: Sloan, L., Morgan, J., Burnap, P., & Williams, M. (2015). Who Tweets? Deriving the Demographic Characteristics of Age, Occupation and Social Class from Twitter User Meta-Data. PLOS ONE, 10(3), e0115545. doi:10.1371/journal.pone.0115545.

# Contact

For any problem you can contact me at : tim.faverjon@sciencespo.fr
