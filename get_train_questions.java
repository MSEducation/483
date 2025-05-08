import java.util.Random;

public class Main {
	public static void main(String[] args) {
		// print train data split
		String[] n = new String[5];
		n[0] = "The dominant paper in our nation's capital, it's among the top 10 U.S. papers in circulation";
		n[1] = "Daniel Hertzberg & James B. Stewart of this paper shared a 1988 Pulitzer for their stories about insider trading";
		n[2] = "Early in their careers, Mark Twain & Bret Harte wrote pieces for this California city's Chronicle";
		n[3] = "In 1840 Horace Greeley began publishing \"The Log Cabin\", a weekly campaign paper in support of this Whig candidate";
		n[4] = "This Georgia paper is known as the AJC for short";
		fisherYates(n);
		System.out.println("NEWSPAPERS");
		System.out.println(n[0]);
		System.out.println(n[1]);
		String[] o = new String[4];
		o[0] = "The practice of pre-authorizing presidential use of force dates to a 1955 resolution re: this island near mainland China";
		o[1] = "The Kentucky & Virginia resolutions were passed to protest these controversial 1798 acts of Congress";
		o[2] = "Feb. 1, National Freedom Day, is the date in 1865 when a resolution sent the states an amendment ending this";
		o[3] = "U.N. Res. 242 supports \"secure and recognized boundaries\" for Israel & neighbors following this June 1967 war";
		fisherYates(o);
		System.out.println("OLD YEAR'S RESOLUTIONS");
		System.out.println(o[0]);
		System.out.println(o[1]);
		String[] b = new String[1];
		b[0] = "Song that says, \"you make me smile with my heart; your looks are laughable, unphotographable\"";
		fisherYates(b);
		String[] p = new String[3];
		p[0] = "In 2011 bell ringers for this charity started accepting digital donations to its red kettle";
		p[1] = "One of the N.Y. Times' headlines on this landmark 1973 Supreme Court decision was \"Cardinals shocked\"";
		p[2] = "Large specimens of this marsupial can leap over barriers 6 feet high";
		fisherYates(p);
		System.out.println("POTPOURRI");
		System.out.println(p[0]);
		String[] s = new String[5];
		//s[0] = 
		// NOTE: I changed my mind about writing this so I just executed it how it is now, took the questions that I got so far, and then got the rest of the train-split questions using the Fisher-Yates algorithm on this website: https://www.gigacalculator.com/randomizers/randomizer.php
		/*HashMap<String, String[]> m = Map.of(
			"NEWSPAPERS", n,
			"OLD YEAR'S RESOLUTIONS", o,
			"BROADWAY LYRICS", b,
			"POTPOURRI", [],
			"STATE OF THE ART MUSEUM (Alex: We'll give you the museum. You give us the state.)", [],
			"\"TIN\" MEN", [],
			"UCLA CELEBRITY ALUMNI", [],
			"SERVICE ORGANIZATIONS", [],
			"AFRICAN CITIES", [],
			"HISTORICAL QUOTES", [],
			"CEMETERIES", [],
			"GOLDEN GLOBE WINNERS", [],
			"HISTORICAL HODGEPODGE", [],
			"CONSERVATION", [],
			"'80s NO.1 HITMAKERS", [],
			"AFRICAN-AMERICAN WOMEN", [],
			"1920s NEWS FLASH!", [],
			"CAMBODIAN HISTORY & CULTURE", [],
			"HE PLAYED A GUY NAMED JACK RYAN IN...", [],
			"I'M BURNIN' FOR YOU", [],
			"NAME THE PARENT COMPANY", [],
			"THE RESIDENTS", [],
			"NOTES FROM THE CAMPAIGN TRAIL", [],
			"POETS & POETRY", [],
			"CAPITAL CITY CHURCHES (Alex: We'll give you the church. You tell us the capital city in which it is located.)", [],
			"THAT 20-AUGHTS SHOW", [],
			"THE QUOTABLE KEATS", [],
			"GREEK FOOD & DRINK", [],
			"RANKS & TITLES", [],
			"COMPLETE DOM-INATION(Alex: Not \"domination.\")", []
		);*/
	}

	private static Random random = new Random(2025);
	// Fisher-Yates shuffle algorithm implementation source: https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
	private static void fisherYates(String[] a) {
		int n = a.length;
		for (int i = n-1; i > 0; i--) {
			int j = random.nextInt(i+1);
			String tmp = a[i];
			a[i] = a[j];
			a[j] = tmp;
		}
	}
}
