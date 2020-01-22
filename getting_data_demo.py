import bs4, requests, re, csv

## function to retrieve the data from an url
def get_data_from_url(url):
    page = requests.get(url)
    all_data = page.text
    return bs4.BeautifulSoup(all_data, features="lxml")

## function to get the main paragraphs texts from the p tags
def get_main_paragraphs(soup):
    all_ps = [str(p) for p in soup.find_all('p')]
    main_ps = [p for p in all_ps if p.startswith('<p>')]
    main_ps =[re.sub(r'<.+?>', '', p) for p in main_ps]
    return ' '.join(main_ps)

## urls for the irish_rugby/positive class... you need more more for your project
monkey_urls = [
    "https://www.newscientist.com/article/2195890-unique-chimpanzee-cultures-are-disappearing-thanks-to-humans/",
    "https://www.newscientist.com/article/2189236-black-haired-monkeys-in-costa-rica-are-suddenly-turning-blonde/",
    "https://www.newscientist.com/article/2185110-an-extinct-monkey-evolved-to-live-like-a-sloth-in-the-caribbean/",
    "https://www.newscientist.com/article/2187943-monkeys-chill-out-just-from-seeing-their-friends-being-groomed/",
    "https://www.newscientist.com/article/2169354-ape-midwives-spotted-helping-female-bonobos-give-birth/",
    "https://www.newscientist.com/article/2168162-bonobos-barely-use-their-opposable-thumbs-when-climbing-trees/",
    "https://www.newscientist.com/article/2170005-chimp-evolution-was-shaped-by-sex-with-their-bonobo-relatives/",
    "https://www.newscientist.com/article/mg23731630-600-primate-archaeology-digging-up-secrets-of-the-monkey-stone-age/",
    "https://www.newscientist.com/article/mg23731623-400-what-does-chinas-monkey-breakthrough-mean-for-human-cloning/",
    "https://www.newscientist.com/article/mg23731623-600-scientists-have-cloned-monkeys-and-it-could-help-treat-cancer/",
    "https://www.newscientist.com/article/2159509-chimps-are-now-dying-of-the-common-cold-and-they-are-all-at-risk/",
    "https://www.newscientist.com/article/2150258-male-chimpanzee-seen-snatching-seconds-old-chimp-and-eating-it/sh-to-edge-of-extinction/",
    "https://www.newscientist.com/article/2152868-watch-a-monkey-floss-its-teeth-with-a-bird-feather/",
    "https://www.newscientist.com/article/2145338-low-ranked-female-monkeys-band-together-against-their-leaders/",
    "https://www.newscientist.com/article/2144440-grown-up-chimps-are-less-likely-to-help-distressed-friends/",
    "https://www.newscientist.com/article/2142389-signs-of-alzheimers-found-in-chimpanzees-for-the-first-time/",
    "https://www.newscientist.com/article/2129077-chimps-pass-on-sponge-drinking-trick-like-a-family-tradition/",
    "https://www.newscientist.com/article/2120476-being-friendly-puts-monkeys-at-risk-in-times-of-revolution/",
    "https://www.newscientist.com/article/2119677-chimps-beat-up-murder-and-then-cannibalise-their-former-tyrant/",
    "https://www.newscientist.com/article/mg23331101-200-we-need-smarter-ways-to-probe-primate-brains/",
    "https://www.newscientist.com/article/2118359-majority-of-primate-species-may-vanish-in-next-25-to-50-years/",
    "https://www.newscientist.com/article/2119610-yellow-fever-outbreak-is-killing-off-rare-monkeys-in-brazil/",
    "https://www.newscientist.com/article/2117731-baboons-recorded-making-key-sounds-found-in-human-speech/",
    "https://www.newscientist.com/article/2117589-wild-monkey-filmed-mounting-deer-and-trying-to-have-sex-with-it/",
    "https://www.newscientist.com/article/2114070-being-popular-is-good-for-health-in-monkeys-at-least/"
    ]
## urls for the not_irish_rugby negative class... you need more for your project
other_urls = [
    "http://journalofmusic.com/focus/why-are-we-neglecting-female-tune-composers",
    "http://journalofmusic.com/focus/we-think-you-should-be-conductor-interview-eimear-noone",
    "http://journalofmusic.com/focus/weve-got-one-chance-get-right-interview-anthony-long",
    "http://journalofmusic.com/focus/how-organise-how-build-how-express-interview-kaija-saariaho",
    "http://journalofmusic.com/focus/why-we-should-be-listening-folk-music",
    "http://journalofmusic.com/focus/new-music-margins",
    "https://www.purina.com/articles/dog/care/winter-safety-tips-for-dogs-in-snow",
    "https://www.purina.com/articles/dog/getting-a-dog/how-to-adopt-a-dog",
    "https://www.purina.com/articles/dog/behavior/why-is-my-dog-chewing-everything",
    "https://www.purina.com/articles/dog/nutrition/can-dogs-eat-carrots",
    "https://www.newscientist.com/article/2197008-asteroid-bennu-is-spewing-out-dust-and-rocks-to-create-its-own-moons/",
    "https://www.newscientist.com/article/2196990-asteroid-ryugu-is-so-dry-we-may-have-to-rethink-how-earth-got-water/",
    "https://www.newscientist.com/article/2196916-distant-space-rock-ultima-thule-formed-in-a-slow-and-gentle-collision/",
    "https://www.newscientist.com/article/2196718-toilet-on-international-space-station-gets-a-bacteria-killing-upgrade/",
    "https://www.newscientist.com/article/2196552-earth-may-be-partly-made-of-rocks-from-elsewhere-in-the-galaxy/",
    "https://www.newscientist.com/article/mg24132210-200-our-wooden-future-making-cars-skyscrapers-and-even-lasers-from-wood/",
    "https://www.newscientist.com/article/mg24132213-700-russian-military-is-building-a-flying-vehicle-with-rotating-paddles/",
    "https://www.newscientist.com/article/mg24132213-500-handheld-device-could-detect-crispr-bioweapons-before-they-spread/",
    "https://www.newscientist.com/article/mg24132202-300-your-5g-guide-will-we-all-benefit-from-super-quick-mobile-internet/h-robot-subs/",
    "https://www.newscientist.com/article/2196117-uk-and-other-eu-countries-ban-boeing-737-max-after-ethiopia-crash/",
    "https://www.purina.com/articles/dog/care/6-tips-how-to-keep-dogs-cool",
    "https://www.purina.com/articles/cat/feeding/what-do-cats-eat",
    "https://www.purina.com/articles/cat/nutrition/can-cats-eat-eggs",
    "https://www.newscientist.com/article/mg24132210-100-too-much-sunscreen-why-avoiding-the-sun-could-damage-your-health/",
    "https://www.newscientist.com/article/mg24132190-700-im-travelling-the-world-to-collect-poo-for-the-good-of-humankind/"
]

## get the data/soups
monkey_soups = [get_data_from_url(url) for url in monkey_urls]
other_soups = [get_data_from_url(url) for url in other_urls]

## retrieve main paragraphs'texts for both types of docs
monkey_docs = [get_main_paragraphs(soup) for soup in monkey_soups]
other_docs = [get_main_paragraphs(soup) for soup in other_soups]

## put the data into a list of dictionaries
data = [{'label':'monkey','doc': doc} for doc in monkey_docs] +  [{'label':'not_monkey','doc': doc} for doc in other_docs]

##write the data to a csv file
with open('raw_data.csv', 'w', newline='', encoding='utf-8') as csvFile:
    attributes = ['label', 'doc']
    writer = csv.DictWriter(csvFile, fieldnames=attributes)
    writer.writeheader()
    writer.writerows(data)
csvFile.close()
