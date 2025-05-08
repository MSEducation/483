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
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

public class IndexFiles {
	public static void main(String[] args) throws IOException {
		Analyzer analyzer = new EnglishAnalyzer();
		IndexWriterConfig config = new IndexWriterConfig(analyzer);
		config.setOpenMode(OpenMode.CREATE);
		// Do not normalize on document length, because Wikipedia articles will not try to cheat an IR system by artificially increasing the length of the articles.
		// We can also consider increasing the importance of term frequency on the returned results because we can trust that Wikipedia articles won't artificially repeat terms just to cheat an IR system. The question is how much we might want to increase the importance of term frequency though. I'm going to set it to 3.0f, which is higher than Lucene's default of 1.2f for BM25Similarity. Using 3.0f means that term repetition will increase a document's ranking, which seems intuitively correct to me. Hopefully 3.0f is not too high. I will leave the discountOverlaps=true because that is what Lucene has as the default.
		Similarity similarity = new BM25Similarity(3.0f, 0.0f, true);
		config.setSimilarity(similarity);
		Directory directory = FSDirectory.open(Paths.get("index"));
		IndexWriter writer = new IndexWriter(directory, config);
		Files.walkFileTree(
			Paths.get("../prepared-wiki"),
			new SimpleFileVisitor<>() {
				@Override
				public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
					try {
						//if (attrs.size() > 5_000) { 
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

								//if (demoEmbeddings != null) {
								//	try (InputStream in = Files.newInputStream(file)) {
								//		float[] vector =
								//			demoEmbeddings.computeEmbedding(
								//					new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8)));
								//		doc.add(
								//				new KnnFloatVectorField(
								//					"contents-vector", vector, VectorSimilarityFunction.DOT_PRODUCT));
								//	}
								//}

								writer.addDocument(document);
							}
						//}
					} catch (IOException e) {}
					return FileVisitResult.CONTINUE;
				}
			}
		);
		writer.close();

		/*directory = FSDirectory.open(Paths.get("index"));
		try (IndexReader reader = DirectoryReader.open(directory)) {
        System.out.println(
            "Indexed "
                + reader.numDocs()
                + " documents in "
                + " ms");
		}*/
		IndexReader reader = DirectoryReader.open(directory);
		IndexSearcher searcher = new IndexSearcher(reader);
		searcher.setSimilarity(similarity);
		analyzer = new EnglishAnalyzer();
		SimpleQueryParser parser = new SimpleQueryParser(analyzer, "contents");
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
		int correct = 0;
		double rrSum = 0.0; // reciprocal rank sum
		for (int i = 0; i < train.length; i++) {
			String expectedAnswer = train[i][1];
			Query query = parser.parse(train[i][0]);
			TopDocs results = searcher.search(query, 5);
			ScoreDoc[] hits = results.scoreDocs;
			StoredFields storedFields = searcher.storedFields();
    			int hitsToEvaluateNum = Math.toIntExact(results.totalHits.value());
			hitsToEvaluateNum = Math.min(hitsToEvaluateNum, 5); 
			if (hitsToEvaluateNum > 0) {
				String proposedAnswer = storedFields.document(hits[0].doc).get("answer");
				if (proposedAnswer.equals(expectedAnswer)) {
					correct++;
					rrSum++;
					System.out.println("Correct: " + proposedAnswer);
				} else {
					System.out.println("Incorrect: " + proposedAnswer + " and Expected: " + expectedAnswer + " (there may be alternatives that are considered correct!)");
					// no need to check when j=0, since we already did that above.
					for (int j = 1; j < hitsToEvaluateNum; j++) {
						proposedAnswer = storedFields.document(hits[j].doc).get("answer");
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
		reader.close();
	}
}
