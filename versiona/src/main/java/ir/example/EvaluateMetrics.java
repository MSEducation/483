package ir.example;
// I started with the Lucene Demo project source code and modified it to arrive here.

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KeywordField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
//import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.IOUtils;
import java.util.ArrayList;
import java.util.List;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.chat.prompt.PromptTemplate;
import java.util.Map;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/knn/DemoEmbeddings.java
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;

// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/knn/KnnVectorDictFilter.java
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.VectorUtil;

// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/knn/KnnVectorDict.java
import static org.apache.lucene.util.fst.FST.readMetadata;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.regex.Pattern;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.PositiveIntOutputs;
import org.apache.lucene.util.fst.Util;

@SpringBootApplication
public class EvaluateMetrics {
	public static void main(String[] args) throws IOException {
		SpringApplication.run(EvaluateMetrics.class, args);

		Analyzer analyzer = new EnglishAnalyzer();
		IndexWriterConfig config = new IndexWriterConfig(analyzer);
		config.setOpenMode(OpenMode.CREATE);
		Directory directory = FSDirectory.open(Paths.get("index"));

		// Calculate embedding vectors for KnnVector search
		KnnVectorDict.build(Paths.get("glove-knndict"), directory, "knn-dict");
		KnnVectorDict vectorDict = new KnnVectorDict(directory, "knn-dict");
		long vectorDictSize = vectorDict.ramBytesUsed();
		FactoryOfEmbeddingForUseInKnn embeddingFactory = new FactoryOfEmbeddingForUseInKnn(vectorDict);

		Similarity similarity = new BM25Similarity();
		config.setSimilarity(similarity);
		IndexWriter writer = new IndexWriter(directory, config);
		Files.walkFileTree(
			Paths.get("../cleaned-prepared-wiki"),
			new SimpleFileVisitor<>() {
				@Override
				public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
					try {
						// The correct answer to a Jeopardy! clue is unlikely to be a Wikipedia article with little information. So, let's help our Lucene IR engine not get confused/mislead by small documents. We do this be simply not indexing documents that are 5000 bytes or less in size.
						if (attrs.size() > 5_000) { 
							BufferedReader reader = new BufferedReader(new FileReader(file.toString()));
							String docTitle = reader.readLine();
							docTitle = docTitle.substring(2, docTitle.length()-2); // get rid of [[ and ]] from the document title.
							reader.close();
							try (InputStream stream = Files.newInputStream(file)) {
								Document document = new Document();

								// Add the Wikipedia title as a field named "answer".  Use a
								// field that is indexed (i.e. searchable), but don't tokenize
								// the field into separate words and don't index term frequency
								// or positional information:
								document.add(new KeywordField("answer", docTitle, Field.Store.YES));

								document.add(new TextField("contents", new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))));

								try (InputStream in = Files.newInputStream(file)) {
									float[] vector =
										embeddingFactory.computeEmbedding(
												new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8)));
									document.add(
											new KnnFloatVectorField(
												"contents-vector", vector, VectorSimilarityFunction.DOT_PRODUCT));
								}

								writer.addDocument(document);
							}
						}
					} catch (IOException e) {}
					return FileVisitResult.CONTINUE;
				}
			}
		);
		writer.close();
		IOUtils.close(vectorDict);

		/*directory = FSDirectory.open(Paths.get("index"));
		try (IndexReader reader = DirectoryReader.open(directory)) {
        System.out.println(
            "Indexed "
                + reader.numDocs()
                + " documents in "
                + " ms");
		}*/
		DirectoryReader reader = DirectoryReader.open(directory);
		IndexSearcher searcher = new IndexSearcher(reader);
		searcher.setSimilarity(similarity);
		analyzer = new EnglishAnalyzer();
		vectorDict = new KnnVectorDict(reader.directory(), "knn-dict");
		SimpleQueryParser parser = new SimpleQueryParser(analyzer, "contents");
		String[][] train = {
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
		int correct = 0;
		double rrSum = 0.0; // reciprocal rank sum
		for (int i = 0; i < train.length; i++) {
			String expectedAnswer = train[i][1];
			Query query = parser.parse(train[i][0]);
			query = addSemanticQuery(query, vectorDict, 5); // knnVectors=5
			TopDocs results = searcher.search(query, 5); // number of results to return = 5
			String[] reranked = rerankWithLlm(searcher, results, train[i][0]);
			//System.out.println("RERANKED IS: " + reranked);
    			int hitsToEvaluateNum = Math.toIntExact(results.totalHits.value());
			hitsToEvaluateNum = Math.min(hitsToEvaluateNum, 5); 
			int rerankedToEvaluateNum = Math.min(reranked.length, hitsToEvaluateNum);
			if (rerankedToEvaluateNum > 0) {
				String proposedAnswer = reranked[0];
				if (proposedAnswer.equals(expectedAnswer)) {
					correct++;
					rrSum++;
					System.out.println("Correct: " + proposedAnswer);
				} else {
					System.out.println("Incorrect: " + proposedAnswer + " and Expected: " + expectedAnswer + " (there may be alternatives that are considered correct!)");
					// no need to check when j=0, since we already did that above.
					for (int j = 1; j < rerankedToEvaluateNum; j++) {
						proposedAnswer = reranked[j];
						if (proposedAnswer.equals(expectedAnswer)) {
							rrSum += 1.0/(1.0+j);
							System.out.println("Correct: " + proposedAnswer);
							break;
						} else {
							System.out.println("Incorrect: " + proposedAnswer + " and Expected: " + expectedAnswer + " (there may be alternatives that are considered correct!)");
						}
					}
					// If the answer was nowhere to be found, rrSum is correctly not incremented.
				}
			}
		}
		System.out.println("Precision@1: " + (((double)correct)/train.length));
		System.out.println("MRR: " + (rrSum/train.length));
		//reader.close();
		IOUtils.close(vectorDict, reader);
	}

	private static String[] rerankWithLlm(IndexSearcher searcher, TopDocs atMostFiveResults, String query) throws IOException {
		StringBuffer possibilities = new StringBuffer();
		ScoreDoc[] hits = atMostFiveResults.scoreDocs;
		StoredFields storedFields = searcher.storedFields();
		int hitsToEvaluateNum = Math.toIntExact(atMostFiveResults.totalHits.value());
		hitsToEvaluateNum = Math.min(hitsToEvaluateNum, 5); 
		if (hitsToEvaluateNum == 0) {
			return new String[5];
		}
		for (int j = 0; j < hitsToEvaluateNum; j++) {
			possibilities.append(storedFields.document(hits[j].doc).get("answer"));
			possibilities.append("\n");
		}

		OllamaOptions options = new OllamaOptions();
		options.setModel("llama3");
		options.setTemperature(0.0);
		options.setTopP(0.0);
		options.setTopK(5);
		options.setSeed(2025);
		OllamaChatModel chatClient = OllamaChatModel.builder().ollamaApi(new OllamaApi("http://localhost:11434")).defaultOptions(options).build();
		//System.out.println(chatClient.call(new Prompt(possibilities.toString(), options)).getResult().getOutput().toString());

		String promptMessage = """
			The following are possible answers:
			{possibilities}

			Reply with only a comma-separated list, nothing more. Rank the aforementioned possible answers into a comma-separated list of 5 possible answers ranked from the best possible answer to the worst possible answer for this clue: {clue}
		""";

		String output = chatClient.call(new PromptTemplate(promptMessage, Map.of("possibilities",possibilities,"clue",query)).create()).getResult().getOutput().getText();
		System.out.println("LLM OUTPUT IS: " + output);

		String[] reranked = output.split(",");
		for (int j = 0; j < reranked.length; j++) {
			reranked[j] = reranked[j].trim(); // In case the LLM puts spaces after the commas.
		}
		return reranked;
		//String[] reranked = new String[5]; // it is okay if we dont have exactly 5 docs to put in here. The end of the array will just not be used by this method's caller.
		//for (int j = 0; j < hitsToEvaluateNum; j++) {
		//	possibilities.append("\n");
		//}

	}
	
	// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/SearchFiles.java
	private static Query addSemanticQuery(Query query, KnnVectorDict vectorDict, int k)
			throws IOException {
			StringBuilder semanticQueryText = new StringBuilder();
			QueryFieldTermExtractor termExtractor = new QueryFieldTermExtractor("contents");
			query.visit(termExtractor);
			for (String term : termExtractor.terms) {
				semanticQueryText.append(term).append(' ');
			}
			if (semanticQueryText.length() > 0) {
				KnnFloatVectorQuery knnQuery =
					new KnnFloatVectorQuery(
							"contents-vector",
							new FactoryOfEmbeddingForUseInKnn(vectorDict).computeEmbedding(semanticQueryText.toString()),
							k);
				BooleanQuery.Builder builder = new BooleanQuery.Builder();
				builder.add(query, BooleanClause.Occur.SHOULD);
				builder.add(knnQuery, BooleanClause.Occur.SHOULD);
				return builder.build();
			}
			return query;
	}

	private static class QueryFieldTermExtractor extends QueryVisitor {
		private final String field;
		private final List<String> terms = new ArrayList<>();

		QueryFieldTermExtractor(String field) {
			this.field = field;
		}

		@Override
		public boolean acceptField(String field) {
			return field.equals(this.field);
		}

		@Override
		public void consumeTerms(Query query, Term... terms) {
			for (Term term : terms) {
				this.terms.add(term.text());
			}
		}

		@Override
		public QueryVisitor getSubVisitor(BooleanClause.Occur occur, Query parent) {
			if (occur == BooleanClause.Occur.MUST_NOT) {
				return QueryVisitor.EMPTY_VISITOR;
			}
			return this;
		}
	}
}

// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/knn/DemoEmbeddings.java
class FactoryOfEmbeddingForUseInKnn {
	private final Analyzer analyzer;

	public FactoryOfEmbeddingForUseInKnn(KnnVectorDict vectorDict) {
		analyzer = new Analyzer() {
			@Override
			protected TokenStreamComponents createComponents(String fieldName) {
				Tokenizer tokenizer = new StandardTokenizer();
				TokenStream output =
					new KnnVectorDictFilter(new LowerCaseFilter(tokenizer), vectorDict);
				return new TokenStreamComponents(tokenizer, output);
			}
		};
	}

	public float[] computeEmbedding(String input) throws IOException {
		return computeEmbedding(new StringReader(input));
	}

	public float[] computeEmbedding(Reader input) throws IOException {
		try (TokenStream tokens = analyzer.tokenStream("dummyField", input)) {
			tokens.reset();
			while (tokens.incrementToken()) {}
			tokens.end();
			return ((KnnVectorDictFilter) tokens).getResult();
		}
	}
}

// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/knn/KnnVectorDictFilter.java
final class KnnVectorDictFilter extends TokenFilter {
	private final KnnVectorDict dict;
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
	private final float[] scratchFloats;
	private final float[] result;
	private final byte[] scratchBytes;
	private final FloatBuffer scratchBuffer;

	public KnnVectorDictFilter(TokenStream input, KnnVectorDict dict) {
		super(input);
		this.dict = dict;
		result = new float[dict.getDimension()];
		scratchBytes = new byte[dict.getDimension() * Float.BYTES];
		scratchBuffer = ByteBuffer.wrap(scratchBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
		scratchFloats = new float[dict.getDimension()];
	}

	@Override
	public boolean incrementToken() throws IOException {
		if (input.incrementToken() == false) {
			return false;
		}
		BytesRef term = new BytesRef(termAtt.toString());
		dict.get(term, scratchBytes);
		scratchBuffer.position(0);
		scratchBuffer.get(scratchFloats);
		VectorUtil.add(result, scratchFloats);
		return true;
	}

	@Override
	public void reset() throws IOException {
		super.reset();
		Arrays.fill(result, 0);
	}

	@Override
	public void end() throws IOException {
		super.end();
		VectorUtil.l2normalize(result, false);
	}

	public float[] getResult() {
		return result;
	}
}

// Source: https://github.com/apache/lucene/blob/main/lucene/demo/src/java/org/apache/lucene/demo/knn/KnnVectorDict.java
class KnnVectorDict implements Closeable {

	private final FST<Long> fst;
	private final IndexInput vectors;
	private final int dimension;

	public KnnVectorDict(Directory directory, String dictName) throws IOException {
		try (IndexInput fstIn = directory.openInput(dictName + ".fst", IOContext.DEFAULT)) {
			fst = new FST<>(readMetadata(fstIn, PositiveIntOutputs.getSingleton()), fstIn);
		}

		vectors = directory.openInput(dictName + ".bin", IOContext.DEFAULT);
		long size = vectors.length();
		vectors.seek(size - Integer.BYTES);
		dimension = vectors.readInt();
		if ((size - Integer.BYTES) % (dimension * (long) Float.BYTES) != 0) {
			throw new IllegalStateException(
					"vector file size " + size + " is not consonant with the vector dimension " + dimension);
		}
	}

	public void get(BytesRef token, byte[] output) throws IOException {
		if (output.length != dimension * Float.BYTES) {
			throw new IllegalArgumentException(
					"the output array must be of length "
					+ (dimension * Float.BYTES)
					+ ", got "
					+ output.length);
		}
		Long ord = Util.get(fst, token);
		if (ord == null) {
			Arrays.fill(output, (byte) 0);
		} else {
			vectors.seek(ord * dimension * Float.BYTES);
			vectors.readBytes(output, 0, output.length);
		}
	}

	public int getDimension() {
		return dimension;
	}

	@Override
	public void close() throws IOException {
		vectors.close();
	}

	public static void build(Path gloveInput, Directory directory, String dictName)
			throws IOException {
			new Builder().build(gloveInput, directory, dictName);
	}

	private static class Builder {
		private static final Pattern SPACE_RE = Pattern.compile(" ");

		private final IntsRefBuilder intsRefBuilder = new IntsRefBuilder();
		private final FSTCompiler<Long> fstCompiler;
		private float[] scratch;
		private ByteBuffer byteBuffer;
		private long ordinal = 1;
		private int numFields;

		Builder() throws IOException {
			fstCompiler =
				new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, PositiveIntOutputs.getSingleton())
				.build();
		}

		void build(Path gloveInput, Directory directory, String dictName) throws IOException {
			try (BufferedReader in = Files.newBufferedReader(gloveInput);
					IndexOutput binOut = directory.createOutput(dictName + ".bin", IOContext.DEFAULT);
					IndexOutput fstOut = directory.createOutput(dictName + ".fst", IOContext.DEFAULT)) {
				writeFirstLine(in, binOut);
				while (addOneLine(in, binOut)) {
					// continue;
				}
				FST.fromFSTReader(fstCompiler.compile(), fstCompiler.getFSTReader()).save(fstOut, fstOut);
				binOut.writeInt(numFields - 1);
					}
		}

		private void writeFirstLine(BufferedReader in, IndexOutput out) throws IOException {
			String[] fields = readOneLine(in);
			if (fields == null) {
				return;
			}
			numFields = fields.length;
			byteBuffer =
				ByteBuffer.allocate((numFields - 1) * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
			scratch = new float[numFields - 1];
			writeVector(fields, out);
		}

		private String[] readOneLine(BufferedReader in) throws IOException {
			String line = in.readLine();
			if (line == null) {
				return null;
			}
			return SPACE_RE.split(line, 0);
		}

		private boolean addOneLine(BufferedReader in, IndexOutput out) throws IOException {
			String[] fields = readOneLine(in);
			if (fields == null) {
				return false;
			}
			if (fields.length != numFields) {
				throw new IllegalStateException(
						"different field count at line "
						+ ordinal
						+ " got "
						+ fields.length
						+ " when expecting "
						+ numFields);
			}
			fstCompiler.add(Util.toIntsRef(new BytesRef(fields[0]), intsRefBuilder), ordinal++);
			writeVector(fields, out);
			return true;
		}

		private void writeVector(String[] fields, IndexOutput out) throws IOException {
			byteBuffer.position(0);
			FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
			for (int i = 1; i < fields.length; i++) {
				scratch[i - 1] = Float.parseFloat(fields[i]);
			}
			VectorUtil.l2normalize(scratch);
			floatBuffer.put(scratch);
			byte[] bytes = byteBuffer.array();
			out.writeBytes(bytes, bytes.length);
		}
	}

	public long ramBytesUsed() {
		return fst.ramBytesUsed() + vectors.length();
	}
}
