import java.util.HashSet;
import java.util.Set;

public class Main {
	public static void main(String[] args) {
		String[][] train = {
                        {"In 1840 Horace Greeley began publishing \"The Log Cabin\", a weekly campaign paper in support of this Whig candidate", "William Henry Harrison"},
                        {"Early in their careers, Mark Twain & Bret Harte wrote pieces for this California city's Chronicle", "San Francisco"},
                        {"The practice of pre-authorizing presidential use of force dates to a 1955 resolution re: this island near mainland China", "Taiwan"},
                        {"U.N. Res. 242 supports \"secure and recognized boundaries\" for Israel & neighbors following this June 1967 war", "The Six Day War"},
                        {"In 2011 bell ringers for this charity started accepting digital donations to its red kettle", "The Salvation Army"},
                        {"The Sun Valley Center for the Arts", "Idaho"},
                        {"The Kalamazoo Institute of Arts", "Michigan"},
                        {"This Italian painter depicted the \"Adoration of the Golden Calf\"", "Tintoretto"},
                        {"He served in the KGB before becoming president & then prime minister of Russia", "Vladimir Putin"},
                        {"Neurobiologist Amy Farrah Fowler on \"The Big Bang Theory\", in real life she has a Ph.D. in neuroscience from UCLA", "Mayim Bialik"},
                        {"This blonde beauty who reprised her role as Amanda on the new \"Melrose Place\" was a psychology major", "Heather Locklear"},
                        {"Originally this club's emblem was a wagon wheel; now it's a gearwheel with 24 cogs & 6 spokes", "Rotary International"},
                        {"This port is the southernmost of South Africa's 3 capitals", "Cape Town"},
                        {"The name of this largest Moroccan city combines 2 Spanish words", "Casablanca"},
                        {"After the fall of France in 1940, this general told his country, \"France has lost a battle. But France has not lost the war\"", "Charles de Gaulle"},
                        {"The mast from the USS Maine is part of the memorial to the ship & crew at this national cemetery", "Arlington National Cemetery"},
                        {"In 2001: The president of the United States on television", "Martin Sheen"},
                        {"In 2009: Joker on film", "Heath Ledger"},
                        {"In the 400s B.C. this Chinese philosopher went into exile for 12 years", "Confucius"},
                        {"The Ammonites held sway in this Mideast country in the 1200s B.C. & the capital is named for them", "Jordan"},
                        {"Indonesia's largest lizard, it's protected from poachers, though we wish it could breathe fire to do the job itself", "Komodo dragon"},
                        {"1980: \"Rock With You\"", "Michael Jackson"},
                        {"1988: \"Man In The Mirror\"", "Michael Jackson"},
                        {"In an essay defending this 2011 film, Myrlie Evers-Williams said, \"My mother was\" this film \"& so was her mother\"", "The Help"},
                        {"Bessie Coleman, the first black woman licensed as a pilot, landed a street named in her honor at this Chicago airport", "O'Hare International Airport"},
                        {"News flash! This less-than-yappy pappy is sixth veep to be nation's top dog after chief takes deep sleep!", "Calvin Coolidge"},
                        {"1922: It's the end of an empire! This empire, in fact! After 600 years, it's goodbye, this, hello, Turkish Republic!", "Ottoman Empire"},
                        {"Not to be confused with karma, krama is a popular accessory sold in cambodia; the word means \"scarf\" in this national language of Cambodia", "Khmer language"},
                        {"\"The Hunt for Red October\"; he went more comedic as Jack Donaghy on \"30 Rock\"", "Alec Baldwin"},
                        {"Pierre Cauchon, Bishop of Beauvais, presided over the trial of this woman who went up in smoke May 30, 1431", "Joan of Arc"},
                        {"Crest toothpaste", "Procter & Gamble"},
                        {"Milton Bradley games", "Hasbro"},
                        {"Don Knotts took over from Norman Fell as the resident landlord on this sitcom", "Three's Company"},
                        {"In \"The Deadlocked Election of 1800\", James R. Sharp outlines the fall of this dueling vice president", "Aaron Burr"},
                        {"One of his \"Tales of a Wayside Inn\" begins, \"Listen, my children, and you shall hear of the midnight ride of Paul Revere\"", "Henry Wadsworth Longfellow"},
                        {"The High Kirk of St. Giles, where John Knox was minister", "Edinburgh"},
                        {"In an 1819 letter Keats wrote that this lord & poet \"cuts a figure, but he is not figurative\"", "Lord Byron"},
                        {"This clear Greek liqueur is quite potent, so it's usually mixed with water, which turns it white & cloudy", "Ouzo"},
                        {"This person is the queen's representative in Canada; currently the office is held by David Johnston", "Governor General of Canada"},
                        {"This New Orleans venue reopened Sept. 25, 2006", "Mercedes-Benz Superdome"},
                };
		String[][] all = {
                        {"NEWSPAPERS The dominant paper in our nation's capital, it's among the top 10 U.S. papers in circulation", "The Washington Post"},
                        {"OLD YEAR'S RESOLUTIONS The practice of pre-authorizing presidential use of force dates to a 1955 resolution re: this island near mainland China", "Taiwan"},
                        {"NEWSPAPERS Daniel Hertzberg & James B. Stewart of this paper shared a 1988 Pulitzer for their stories about insider trading", "The Wall Street Journal"},
                        {"BROADWAY LYRICS Song that says, \"you make me smile with my heart; your looks are laughable, unphotographable\"", "My Funny Valentine"},
                        {"POTPOURRI In 2011 bell ringers for this charity started accepting digital donations to its red kettle", "The Salvation Army"},
                        {"STATE OF THE ART MUSEUM (Alex: We'll give you the museum. You give us the state.) The Naples Museum of Art", "Florida"},
                        {"\"TIN\" MEN This Italian painter depicted the \"Adoration of the Golden Calf\"", "Tintoretto"},
                        {"UCLA CELEBRITY ALUMNI This woman who won consecutive heptathlons at the Olympics went to UCLA on a basketball scholarship", "Jackie Joyner-Kersee"},
                        {"SERVICE ORGANIZATIONS Originally this club's emblem was a wagon wheel; now it's a gearwheel with 24 cogs & 6 spokes", "Rotary International"},
                        {"AFRICAN CITIES Several bridges, including El Tahrir, cross the Nile in this capital", "Cairo"},
                        {"HISTORICAL QUOTES After the fall of France in 1940, this general told his country, \"France has lost a battle. But France has not lost the war\"", "Charles de Gaulle"},
                        {"STATE OF THE ART MUSEUM (Alex: We'll give you the museum. You give us the state.) The Taft Museum of Art", "Ohio"},
                        {"CEMETERIES The mast from the USS Maine is part of the memorial to the ship & crew at this national cemetery", "Arlington National Cemetery"},
                        {"GOLDEN GLOBE WINNERS In 2009: Joker on film", "Heath Ledger"},
                        {"HISTORICAL HODGEPODGE It was the peninsula fought over in the peninsular war of 1808 to 1814", "Iberian Peninsula"},
                        {"CONSERVATION In 1980 China founded a center for these cute creatures in its bamboo-rich Wolong Nature Preserve", "Giant panda"},
                        {"'80s NO.1 HITMAKERS 1988: \"Father Figure\"", "George Michael"},
                        {"AFRICAN-AMERICAN WOMEN In an essay defending this 2011 film, Myrlie Evers-Williams said, \"My mother was\" this film \"& so was her mother\"", "The Help"},
                        {"SERVICE ORGANIZATIONS Father Michael McGivney founded this fraternal society for Catholic laymen in 1882", "Knights of Columbus"},
                        {"CONSERVATION Early projects of the WWF, this organization, included work with the bald eagle & the red wolf", "World Wide Fund"},
                        {"CONSERVATION Indonesia's largest lizard, it's protected from poachers, though we wish it could breathe fire to do the job itself", "Komodo dragon"},
                        {"1920s NEWS FLASH! Nov. 28, 1929! This man & his chief pilot Bernt Balchen fly to South Pole! Yowza! You'll be an admirable admiral, sir!", "Richard Byrd"},
                        {"CEMETERIES On May 5, 1878 Alice Chambers was the last person buried in this Dodge City, Kansas cemetery", "Boot Hill"},
                        {"CAMBODIAN HISTORY & CULTURE The Royal Palace grounds feature a statue of King Norodom, who in the late 1800s was compelled to first put his country under the control of this European power; of course, it was sculpted in that country", "France"},
                        {"HISTORICAL HODGEPODGE In the 400s B.C. this Chinese philosopher went into exile for 12 years", "Confucius"},
                        {"AFRICAN-AMERICAN WOMEN Bessie Coleman, the first black woman licensed as a pilot, landed a street named in her honor at this Chicago airport", "O'Hare International Airport"},
                        {"HISTORICAL HODGEPODGE The Ammonites held sway in this Mideast country in the 1200s B.C. & the capital is named for them", "Jordan"},
                        {"HE PLAYED A GUY NAMED JACK RYAN IN... \"The Sum of All Fears\"; he also won a screenwriting Oscar for \"Good Will Hunting\"", "Ben Affleck"},
                        {"POTPOURRI One of the N.Y. Times' headlines on this landmark 1973 Supreme Court decision was \"Cardinals shocked\"", "Roe v. Wade"},
                        {"I'M BURNIN' FOR YOU France's Philip IV--known as \"The Fair\"--had Jacques De Molay, the last Grand Master of this order, burned in 1314", "Knights Templar"},
                        {"STATE OF THE ART MUSEUM (Alex: We'll give you the museum. You give us the state.) The Georgia O'Keeffe Museum", "New Mexico"},
                        {"AFRICAN CITIES The name of this largest Moroccan city combines 2 Spanish words", "Casablanca"},
                        {"NAME THE PARENT COMPANY Jell-O", "Kraft Foods"},
                        {"GOLDEN GLOBE WINNERS 2011: Chicago mayor Tom Kane", "Kelsey Grammer"},
                        {"THE RESIDENTS Title residence of Otter, Flounder, Pinto & Bluto in a 1978 comedy", "Animal House"},
                        {"UCLA CELEBRITY ALUMNI Neurobiologist Amy Farrah Fowler on \"The Big Bang Theory\", in real life she has a Ph.D. in neuroscience from UCLA", "Mayim Bialik"},
                        {"NOTES FROM THE CAMPAIGN TRAIL In \"The Deadlocked Election of 1800\", James R. Sharp outlines the fall of this dueling vice president", "Aaron Burr"},
                        {"\"TIN\" MEN He served in the KGB before becoming president & then prime minister of Russia", "Vladimir Putin"},
                        {"AFRICAN-AMERICAN WOMEN When asked to describe herself, she says first & foremost, she is Malia & Sasha's mom", "Michelle Obama"},
                        {"POETS & POETRY She wrote, \"My candle burns at both ends... but, ah, my foes, and oh, my friends--it gives a lovely light\"", "Edna St. Vincent Millay"},
                        {"CAPITAL CITY CHURCHES (Alex: We'll give you the church. You tell us the capital city in which it is located.) In this Finnish city, the Lutheran Cathedral, also known as Tuomiokirkko", "Helsinki"},
                        {"NAME THE PARENT COMPANY Milton Bradley games", "Hasbro"},
                        {"OLD YEAR'S RESOLUTIONS The Kentucky & Virginia resolutions were passed to protest these controversial 1798 acts of Congress", "The Alien and Sedition Acts"},
                        {"'80s NO.1 HITMAKERS 1983: \"Beat It\"", "Michael Jackson"},
                        {"GOLDEN GLOBE WINNERS In 2009: Sookie Stackhouse", "Anna Paquin"},
                        {"HISTORICAL HODGEPODGE This member of the Nixon & Ford cabinets was born in Furth, Germany in 1923", "Henry Kissinger"},
                        {"CAPITAL CITY CHURCHES (Alex: We'll give you the church. You tell us the capital city in which it is located.) The High Kirk of St. Giles, where John Knox was minister", "Edinburgh"},
                        {"UCLA CELEBRITY ALUMNI For the brief time he attended, he was a rebel with a cause, even landing a lead role in a 1950 stage production", "James Dean"},
                        {"NAME THE PARENT COMPANY Fisher-Price toys", "Mattel"},
                        {"HISTORICAL QUOTES In a 1959 American kitchen exhibit in Moscow, he told Khrushchev, \"In America, we like to make life easier for women\"", "Richard Nixon"},
                        {"POETS & POETRY One of his \"Tales of a Wayside Inn\" begins, \"Listen, my children, and you shall hear of the midnight ride of Paul Revere\"", "Henry Wadsworth Longfellow"},
                        {"NOTES FROM THE CAMPAIGN TRAIL This bestseller about problems on the McCain-Palin ticket became an HBO movie with Julianne Moore", "Game Change"},
                        {"THAT 20-AUGHTS SHOW A 2-part episode of \"JAG\" introduced this Mark Harmon drama", "NCIS"},
                        {"AFRICAN CITIES This port is the southernmost of South Africa's 3 capitals", "Cape Town"},
                        {"THE QUOTABLE KEATS Keats was quoting this Edmund Spenser poem when he told Shelley to \"'load every rift' of your subject with ore\"", "The Faerie Queene"},
                        {"THE QUOTABLE KEATS In an 1819 letter Keats wrote that this lord & poet \"cuts a figure, but he is not figurative\"", "Lord Byron"},
                        {"GREEK FOOD & DRINK This clear Greek liqueur is quite potent, so it's usually mixed with water, which turns it white & cloudy", "Ouzo"},
                        {"OLD YEAR'S RESOLUTIONS Feb. 1, National Freedom Day, is the date in 1865 when a resolution sent the states an amendment ending this", "Slavery"},
                        {"RANKS & TITLES This person is the queen's representative in Canada; currently the office is held by David Johnston", "Governor General of Canada"},
                        {"\"TIN\" MEN He earned the \"fifth Beatle\" nickname by producing all of the Beatles' albums", "George Martin"},
                        {"NEWSPAPERS Early in their careers, Mark Twain & Bret Harte wrote pieces for this California city's Chronicle", "San Francisco"},
                        {"POTPOURRI Large specimens of this marsupial can leap over barriers 6 feet high", "Kangaroo"},
                        {"GREEK FOOD & DRINK Because it's cured & stored in brine, this crumbly white cheese made from sheep's milk is often referred to as pickled cheese", "Feta"},
                        {"1920s NEWS FLASH! 1927! Gene Tunney takes a long count in the squared circle but rises to defeat this \"Manassa Mauler\"! Howzabout that!", "Jack Dempsey"},
			{"RANKS & TITLES Italian for \"leader\", it was especially applied to Benito Mussolini", "Duce"},
                        {"STATE OF THE ART MUSEUM (Alex: We'll give you the museum. You give us the state.) The Kalamazoo Institute of Arts", "Michigan"},
                        {"STATE OF THE ART MUSEUM (Alex: We'll give you the museum. You give us the state.) The Sun Valley Center for the Arts", "Idaho"},
                        {"\"TIN\" MEN You can't mention this shortstop without mentioning his double-play associates Evers & Chance", "Joe Tinker"},
                        {"NEWSPAPERS In 1840 Horace Greeley began publishing \"The Log Cabin\", a weekly campaign paper in support of this Whig candidate", "William Henry Harrison"},
                        {"I'M BURNIN' FOR YOU Pierre Cauchon, Bishop of Beauvais, presided over the trial of this woman who went up in smoke May 30, 1431", "Joan of Arc"},
                        {"COMPLETE DOM-INATION(Alex: Not \"domination.\") This Wisconsin city claims to have built the USA's only granite dome", "Madison"},
                        {"NEWSPAPERS This Georgia paper is known as the AJC for short", "The Atlanta Journal-Constitution"},
                        {"AFRICAN CITIES Wooden 2-story verandas in this Liberian capital are an architectural link to the U.S. south", "Monrovia"},
                        {"COMPLETE DOM-INATION(Alex: Not \"domination.\") This New Orleans venue reopened Sept. 25, 2006", "Mercedes-Benz Superdome"},
                        {"HE PLAYED A GUY NAMED JACK RYAN IN... \"The Hunt for Red October\"; he went more comedic as Jack Donaghy on \"30 Rock\"", "Alec Baldwin"},
                        {"AFRICAN-AMERICAN WOMEN Rita Dove titled a collection of poems \"On the Bus with\" this woman", "Rosa Parks"},
                        {"HE PLAYED A GUY NAMED JACK RYAN IN... \"Patriot Games\"; he's had other iconic roles, in space & underground", "Harrison Ford"},
                        {"COMPLETE DOM-INATION(Alex: Not \"domination.\") This sacred structure dates from the late 600's A.D.", "Dome of the Rock"},
                        {"'80s NO.1 HITMAKERS 1988: \"Man In The Mirror\"", "Michael Jackson"},
                        {"CAPITAL CITY CHURCHES (Alex: We'll give you the church. You tell us the capital city in which it is located.) Matthias Church, or Matyas Templom, where Franz Joseph was crowned in 1867", "Budapest"},
                        {"UCLA CELEBRITY ALUMNI Attending UCLA in the '60s, he was no \"Meathead\", he just played one later on television", "Rob Reiner"},
                        {"THE RESIDENTS Kinch, Carter & LeBeau were all residents of Stalag 13 on this TV show", "Hogan's Heroes"},
                        {"1920s NEWS FLASH! News flash! This less-than-yappy pappy is sixth veep to be nation's top dog after chief takes deep sleep!", "Calvin Coolidge"},
                        {"GOLDEN GLOBE WINNERS In 2001: The president of the United States on television", "Martin Sheen"},
                        {"'80s NO.1 HITMAKERS 1989: \"Miss You Much\"", "Janet Jackson"},
                        {"1920s NEWS FLASH! 1922: It's the end of an empire! This empire, in fact! After 600 years, it's goodbye, this, hello, Turkish Republic!", "Ottoman Empire"},
                        {"NAME THE PARENT COMPANY Crest toothpaste", "Procter & Gamble"},
                        {"HISTORICAL QUOTES In 1888 this Chancellor told the Reichstag, \"we Germans fear God, but nothing else in the world\"", "Otto von Bismarck"},
                        {"POETS & POETRY In 1787 he signed his first published poem \"Axiologus\"; axio- is from the Greek for \"worth\"", "William Wordsworth"},
                        {"CAMBODIAN HISTORY & CULTURE Not to be confused with karma, krama is a popular accessory sold in cambodia; the word means \"scarf\" in this national language of Cambodia", "Khmer language"},
                        {"CAMBODIAN HISTORY & CULTURE Phnom Penh's notorious gridlock is circumvented by the nimble tuk-tuk, a motorized taxi that's also known as an auto this, a similar Asian conveyance.", "Rickshaw"},
                        {"'80s NO.1 HITMAKERS 1980: \"Rock With You\"", "Michael Jackson"},
                        {"NOTES FROM THE CAMPAIGN TRAIL The Pulitzer-winning \"The Making of the President 1960\" covered this man's successful presidential campaign", "John F. Kennedy"},
                        {"SERVICE ORGANIZATIONS In 1843 Isaac Dittenhoefer became the first pres. of this Jewish club whose name means \"children of the covenant\"", "B'nai B'rith"},
                        {"THE RESIDENTS Don Knotts took over from Norman Fell as the resident landlord on this sitcom", "Three's Company"},
                        {"OLD YEAR'S RESOLUTIONS U.N. Res. 242 supports \"secure and recognized boundaries\" for Israel & neighbors following this June 1967 war", "The Six Day War"},
                        {"UCLA CELEBRITY ALUMNI This blonde beauty who reprised her role as Amanda on the new \"Melrose Place\" was a psychology major", "Heather Locklear"},
                        {"GREEK FOOD & DRINK The name of this dish of marinated lamb, skewered & grilled, comes from the Greek for \"skewer\" & also starts with \"s\"", "Souvlaki"},
                        {"NAME THE PARENT COMPANY Post-it notes", "3M"},
                        {"GOLDEN GLOBE WINNERS In 2010: As Sherlock Holmes on film", "Robert Downey, Jr."}
                };
		/*String[][] test = {
			{"The dominant paper in our nation's capital, it's among the top 10 U.S. papers in circulation", ""},
			{"Daniel Hertzberg & James B. Stewart of this paper shared a 1988 Pulitzer for their stories about insider trading", ""},
			{"Song that says, \"you make me smile with my heart; your looks are laughable, unphotographable\"", ""},
			{"The Naples Museum of Art", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
			{"", ""},
		};*/
		System.out.println(train.length);
		int wordCountSum = 0;
		int nonstopWordCountSum = 0;
		Set<String> s = new HashSet<>();
		for (int i = 0; i < train.length; i++) {
			String[] words = train[i][0].split(" ");
			wordCountSum += words.length;
			for (int j = 0; j < words.length; j++) {
				String w = words[j].toLowerCase();
				if (!w.equals("the") && !w.equals("a") && !w.equals("an") && !w.equals("of") && !w.equals("with") && !w.equals("in") && !w.equals("on")) {
					nonstopWordCountSum++;
					s.add(w);
				}
			}
			
		}
		System.out.println(wordCountSum/train.length);
		System.out.println(nonstopWordCountSum/train.length);
		System.out.println(s.size()/train.length);
		System.out.println("now for ALL:");
		System.out.println(all.length);
		wordCountSum = 0;
		nonstopWordCountSum = 0;
		s = new HashSet<>();
		for (int i = 0; i < all.length; i++) {
			String[] words = all[i][0].split(" ");
			wordCountSum += words.length;
			for (int j = 0; j < words.length; j++) {
				String w = words[j].toLowerCase();
				if (!w.equals("the") && !w.equals("a") && !w.equals("an") && !w.equals("of") && !w.equals("with") && !w.equals("in") && !w.equals("on")) {
					nonstopWordCountSum++;
					s.add(w);
				}
			}
		}
		System.out.println(wordCountSum/all.length);
		System.out.println(nonstopWordCountSum/all.length);
		System.out.println(s.size()/all.length);
	}
}
