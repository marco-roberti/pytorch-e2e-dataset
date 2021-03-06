# coding=utf-8
import re
from math import floor, ceil

from e2e import E2E, E2ESet


class E2ENames(E2E):
    def __init__(self, root, which_set: E2ESet, vocabulary_class):
        super().__init__(root, which_set, vocabulary_class)
        # Make sure all tokens in new_york are present in the vocabulary.
        for restaurant in new_york:
            self.vocabulary.add_sentence(restaurant)

        # According to its paper, E2E dataset is split into training, validation and testing sets in a 76.5-8.5-15 ratio
        new_york_len = len(new_york)
        train_validation = floor(0.765 * new_york_len)
        validation_testing = ceil(0.85 * new_york_len)
        names = {
            E2ESet.TRAIN: new_york[:train_validation],
            E2ESet.DEV:   new_york[train_validation:validation_testing],
            E2ESet.TEST:  new_york[validation_testing:]
        }

        # Actual replacements
        self.sort()
        num_restaurants = len(names[which_set])
        i_restaurant = 0
        last_mr = self.mr[0]
        for i in range(len(self)):
            mr, ref = self.mr[i], self.ref[i]

            if mr != last_mr:
                i_restaurant = (i_restaurant + 1) % num_restaurants
            last_mr = mr

            mr_str = self.to_string(mr)
            ref_str = self.to_string(ref)

            new_restaurant = names[which_set][i_restaurant]
            old_restaurant = re.match(r'name ?\[ *([A-Za-z ]*[A-Za-z]) *\]', mr_str)[1]

            mr_sub = re.sub(old_restaurant, new_restaurant, mr_str)
            ref_sub = re.sub(old_restaurant, new_restaurant, ref_str)

            self.mr[i] = self.vocabulary.add_sentence(mr_sub)
            self.ref[i] = self.vocabulary.add_sentence(ref_sub)


# noinspection SpellCheckingInspection
new_york = ['Natalino', 'Indian Cafe', 'Manhattan Plaza Cafe', 'The Stanhope', 'Chez Napoleon', 'Au Mandarin',
            'SUSHISAY', 'Brasserie', 'Popover Cafe', '107 West', 'Le Train Bleu', 'Bridge Cafe', 'Indian Oven',
            'Solera', 'Automatic Slim\'s', 'Freddie & Pepper Pizza', 'LUTECE', 'Nanni Il Valletto', 'Mezzogiorno',
            'Nicola Paone', 'Rafaella Ristorante', 'Fishin Eddie', 'Lou G. Siegel', 'Morton\'s of Chicago', 'Jezebel',
            'Yellow Rose Cafe', 'Oceana', 'Grove Street Cafe', 'Neary\'s', 'El Charro', 'Manila',
            'Marnie\'s Noodle Shop', 'Garden Cafe', 'Brasserie des Theatres', 'Bendix Diner', 'Mad.61',
            'Harriet\'s Kitchen', 'Cal\'s', 'La Bouillabaisse', 'Nadine\'s', 'Sal Anthony\'s', 'San Domenico',
            'LA GRENOUILLE', 'LE CHANTILLY', 'Wolf\'s 6th Avenue Delicatessen', 'Mo\'s Caribbean Bar & Grille', 'Cafe',
            'Orleans', 'Bellini by Cipriani', 'Orso', 'Columbia Cottage', 'Hulot\'s', 'Cucina di Pesce', 'Ryoyu',
            'Court of Three Sisters', 'Violeta\'s Mexican Restaurant', 'Walker\'s', 'La Tour d\'Or', 'Souen', 'Hasaki',
            'Zona Rosa', 'The Townhouse', 'Caffe Carciofo', 'AN AMERICAN PLACE', 'Il Gabbiano', 'Mambo Grill',
            'Memphis', 'Boonthai', 'Bistro Cafe', 'Vince and Eddie\'s', 'Poiret', 'Caffe Cielo', 'Keens Chop House',
            'City Cafe', 'El Parador Cafe', 'Wong Kee', 'TSE YANG', 'La Frontera', 'Cafe Andrusha', 'Frutti di Mare',
            'Il Menestrello', 'CHANTERELLE', 'Westside Cafe', 'Balbek', 'Appetito', 'Kan Pai', 'Denino\'s Tavern',
            'Moonrock', 'Tony Roma\'s', 'Ponte\'s', 'The Plaza Oyster Bar', 'Il Tinello', 'I Tre Merli',
            'Mission Grill', 'Mary\'s Restaurant', 'Back Porch', 'L\'Acajou', 'Cafe St. John', 'Frank\'s Restaurant',
            'Le Refuge', 'Veselka', 'Old Homestead', 'HUDSON RIVER CLUB', 'Il Cortile', 'Aria', 'Asia', 'Arqua',
            'Cafe Baci', 'Piccola Venezia', 'Bella Mama', 'Sunny East', 'Cafe des Sports', 'Scoop', 'Il Cantinori',
            'Stingy Lulu\'s', 'The Oak Room and Bar', '44', 'Bright Food Shop', 'New Prospect Cafe (Bklyn.)',
            'The Small Cafe', 'Sido Abu Salim', 'Sichuan Palace', 'Aperitivo', 'Amici Miei', 'Felidia',
            'Cafe Nicholson', 'Gonzalez y Gonzalez', 'TAVERN ON THE GREEN', 'RIVER CAFE 10', 'Cafe Espasol',
            'LE CIRQUE', 'Peter Hilary\'s', 'Planet Hollywood', 'La Vieille Auberge', 'La Rivista', 'Genji', 'BTI',
            'Fiori', 'The Black Sheep', 'Va Bene', 'Cafe Greco', 'Benvenuti Ristorante', 'Koo Koo\'s Bistro',
            'Japonica', 'San Martin\'s', 'L\'Incontro', 'Century Cafe', 'Alcala', 'La Colombe d\'Or', 'Benito I',
            'Yuka', 'Luxe', 'Fine & Schapiro', 'America', 'RUSSIAN TEA ROOM', 'CARMINE\'S', 'Akbar', 'Brio',
            'LE BERNARDIN', 'Kodnoi', '101 Seafood', 'Kom Tang Soot Bul House', 'Nello', 'Le Comptoir', 'Nippon',
            'New World Coffee', 'Les Sans Culottes', 'Cafe Nosidam', 'Thai House Cafe', 'Fourteen (fka Quatorze)',
            'Trattoria Pesce Pasta', 'Picholine', 'T-Rex', 'Paul & Jimmy\'s', 'Il Gattopardo', 'Moroccan Star',
            'Caffe BiondoX', 'Miss Ellie\'s Homesick Bar & Grill', 'Trattoria dell\'Arte', 'Buckaroo\'s', 'OYSTER BAR',
            'Pasta Vicci', 'Banana Cafe', 'Opus II', 'Vix Cafe', 'Nirvana', 'Pietro & Vanessa', 'Wollensky\'s Grill',
            'Ci Vediamo', 'Chez Ma Tante', 'Red Lion', 'Lupe\'s East L.A. Kitchen', 'Sarabeth\'s', 'Pinocchio',
            'PERIYALI', 'Pamir', 'Angelica Kitchen', 'WINDOWS ON THE WORLD', 'Jean Claude', 'DAWAT', 'Ferrara',
            'Tibetan Kitchen', 'Sevilla', 'Puket', 'Ratner\'s', 'Woo Chon', 'Old Bermuda Inn', 'Gingertoon',
            'Josephina', 'Tre Scalini', 'Sharkey\'s', 'The Pie', 'Sammy\'s Roumanian', 'MONTRACHET',
            'John\'s of 12th Street', 'Silver Palace', 'Soleil', 'Bistro 790', 'Cabana Carioca', 'Amir\'s Falafel',
            'El Pollo', 'La Strada', 'La Strada', 'Raymond\'s Cafe', 'Aggie\'s', 'H.S.F.', 'Cedars of Lebanon',
            'Zephyr Grill', 'El Quijote', 'Benihana of Tokyo', 'Zula', 'Coco Lezzone', 'Alison on Dominick Street',
            'Jekyll & Hyde', 'Our Place', 'Etats-Unis', 'Vong and Kwong', 'Mr. Chow', 'Mulholland Drive Cafe',
            'Broadway Diner', 'El Teddy\'s', 'Cafe Pierre', 'R.H. Tugs', 'Palio', 'Sarashina', 'Casa La Femme',
            'Rio Mar', 'Shark Bar', 'Conservatory Cafe', 'Il Nostro', 'Antico Caffee', 'Odessa', 'Carino',
            'Jack\'s Place', 'J. Sung Dynasty', 'HATSUHANA', 'Paper Moon Milano', 'Empire Diner', 'Le Bouchon',
            'One Fifth Avenue', 'Pedro Paramo', 'Santa Fe', 'Zachary\'s', 'Casa Di Pre', 'Nick & Eddie', 'Silverado',
            'National', 'Le Madeleine', 'Cooper\'s Coffee Bar', 'Wally\'s and Joseph\'s', 'Ying', 'Day-O',
            'Museum Cafe', 'SONIA ROSE', 'Mitsukoshi', 'Cafe Luxembourg', 'The Sea Grill', 'Dojo', 'Papaya King',
            'Tang Tang', 'Zucchini', 'Snaps', 'Les Pyrenees', 'Cafe Melville', 'San Giusto', 'Odeon', 'Boogies Diner',
            'La Jumelle', 'Le Steak', 'Swiss Inn', 'Le Pactole', 'Japanese on Hudson', 'PRIMAVERA', 'King Crab',
            'Marion\'s ContinentalRestaurant & Lounge', 'Tai Hong Lau', 'Tirami Su', 'Cafe Europa', 'AQUAVIT',
            'Lucky Strike', 'Acadia Parish', 'Au Cafe', 'Two Eleven', 'Mimi\'s', 'Elephant & Castle', 'Busby\'s',
            'Dish of Salt', 'Mo\' Better Restaurant', 'Firenze', 'La Caridad', 'Claire', 'Rascals', 'Shanghai 1933',
            'Cafe Riazor', 'Le Beaujolais', 'Rosie O\'Grady\'s', 'Mediterraneo', 'Tropica', '9', 'Erminia',
            'Chaz & Wilson\'s Grill', 'Acme Bar & Grill', 'Nawab', 'PETROSSIAN', 'Benito II',
            'Angelo\'s of Mulberry St.', 'Cafe 400', 'La Ripaille', 'Roebling\'s', 'Chikubu', 'Tommaso\'s', 'Letizia',
            'Sette Mezzo', 'Mario\'s', 'Regional Thai Taste', 'Becco', 'Tutta Pasta', 'Joey\'s Paesano',
            'SoHo Kitchen and Bar', 'Taliesin', 'Chelsea Clinton Cafe', 'Havana (fka Victor\'s Cafe)', 'Boxers',
            'Arlecchino', 'Great Shanghai', 'Chiam', 'Big Wong', 'Les Halles', 'Ticino', 'Urban Grill', 'The View',
            'Sistina', 'Urbino', 'Stage Deli', 'Sushiden', 'Patriccio\'s', 'Le Quercy', 'Campagnola', 'Girasole',
            'The Yankee Clipper', 'Temple Bar', 'La Fondue', 'Raphael', 'E.A.T', 'Le Relais', 'Island',
            'El Pote Espasol', 'Vinsanto', 'Tatou', 'PETER LUGERSTEAK HOUSE', 'Lexington Avenue Grill', 'Cesarina',
            'LE REGENCE', 'Lorango', 'Isola', 'Fiorello\'s Roman Cafe', 'Lai Lai West', 'Sofia\'s', 'Kiev',
            'Tang Pavilion', 'The Sumptuary', 'Taci International', 'Jules', 'SPARKS STEAK HOUSE', 'Rosa Mexicano',
            'Broadway Grill', 'Westside Cottage', 'Symposium', 'Wylie\'s Ribs', 'Hourglass Tavern', 'Mappamondo',
            'Pietro\'s', 'Chez Brigitte', 'Hard Rock Cafe', 'Can', 'Wing Wong', 'Harlequin', 'Ye Olde Waverly Inn',
            'Shin\'s', 'Pescatore', 'Nino\'s', 'Thai Taste', 'Book-Friends Cafe', 'Le Bar Bat', 'Courtyard Cafe',
            'Swing Street Cafe', 'Capsouto Freres', 'Whole Wheat \'n Wild Berrys', 'Bell Caffe', 'Donald Sacks',
            'Santerello', 'Felix', 'Argentine Pavilion', 'NoHo Star', 'Capriccio', 'Food Bar', 'B. Smith\'s',
            'The Assembly', 'Dolce', 'Play By Play', 'Estia', 'Boom', 'Jewel of India', 'DishS', 'Pisces',
            'Water\'s Edge', 'Harry\'s at Hanover Square', 'Gabriel\'s', 'Dosanko', 'Voulez-Vous', 'L\'Entrecote',
            'Sloppy Louie\'s', 'Grand Ticino', 'Caffe Buon Gusto', 'Yamaguchi', 'Milano', 'Shabu Tatsu', 'Gus\' Place',
            'Cucina Della Fontana', 'Joe Allen', 'Park Avenue Country Club', 'McDonald\'s', 'Crepes Suzette', 'BOULEY',
            'LA COTE BASQUE', 'Anche Vivolo', 'Health Pub', 'Fanelli\'s Cafe', 'Romeo Salta', 'Lolabelle', 'Le Rivage',
            'Manganaro Grosseria Italiana', 'Barney Greengrass', 'Tenth Avenue Jukebox Bar', 'Cascabel', 'Hideaway',
            'Yura', 'Royal Canadian Pancake', 'Nicola\'s', 'Umeda', 'Hamachi', 'Wilkinson\'s Seafood Cafe',
            'Red River Grill', 'Petaluma', 'Henry\'s End', 'Maurya', 'Ennio and Michael', 'Tempo', 'The Polo', 'Darbar',
            'Good Enough to Eat', 'Il Ponte Vecchio', 'LA CARAVELLE', 'Stick To Your Ribs BBQ',
            'Louisiana Community Bar & Grill', 'Ben Benson\'s', 'Gascogne', 'Tennessee Mountain', 'Orologio',
            'Benny\'s Burritos', 'Dakota Bar & Grill', 'Divino', 'Burger Heaven', 'Miracle Grill', 'Anarchy Cafe',
            'A Pasta Place', 'Atomic Wings', 'Trois Jean', 'The Road House', 'PATSY\'S PIZZA (Bklyn)', 'Rolf\'s',
            'Stephanie\'s', 'Lucy\'s Retired Surfers', 'Buono Tavola', 'Pho Pasteur Vietnam', 'Man Ray',
            'Vittorio Cucina', 'Giambelli', 'Fresch', 'Mesa de Espana', 'Roettelle A.G.', 'Ten Kai', 'Jour et Nuit',
            'La Boheme', 'Windows on India', 'Hunan Fifth Ave.', 'Eighteenth & Eighth', 'Primola', 'China Fun',
            'IL MULINO', 'Ballato\'s', 'Tango', 'Golden Unicorn', 'New Viet Huong', 'Il Vagabondo', 'Old Town Bar',
            'LA RESERVE', 'La Petite Ferme', 'Vucciria', 'Uskudar', 'Thailand Restaurant', 'The Leopard',
            'Barnes & Noble Cafe', 'Time Cafe', 'San Pietro', 'Isabella\'s', 'Houlihan\'s', 'The Algonquin Hotel',
            'Girafe', 'Ahnell', 'Jim McMullen', 'La Mirabelle', 'Raoul\'s', 'Cantina', 'White Horse Tavern',
            'Copeland\'s', 'Volare', 'MARK\'S', 'Caffe Rosso', '103 NYC', 'Canyon Road', 'Kinoko', 'Ecco-La',
            'Hunan Balcony', 'Harley Davidson Cafe', 'Chantal Cafe', 'Il Giglio', 'Kurumazushi',
            'Telephone Bar & Grill', 'Westway Diner', 'Menchanko-tei', 'Russian Samovar', 'McSorley\'s Old Ale House',
            'Minetta Tavern', 'ONE IF BY LAND TIBS', 'The Cloister Cafe', 'Triple Eight Palace', 'Petes\' Place',
            'Bora', '5 & 10 No Exaggeration', 'Fino', 'Omen', 'Po', 'Zucchero', 'Zig Zag', 'Rosemarie\'s', 'Valone\'s',
            'Valone\'s', 'Pete\'s Tavern', 'Marumi', 'Trastevere 83', 'Ossie\'s Table', 'POST HOUSE', 'Shinwa (93)',
            'Mary Ann\'s', 'Inagiku', 'Rosolio', 'Edward Moran Bar & Grill', 'Positano', 'Quatorze Bis', 'Indies',
            'Mr. Fuji\'s Tropicana', 'The Palm Court', 'One Hudson Cafe', 'Ararat Russe', 'Hudson Grill',
            'Caffe Vivaldi', 'Fagiolini', 'Telly\'s Taverna', 'Cite Grille', '20 Mott Street', 'Via Oretto',
            'La Spaghetteria', 'Sirabella\'s', 'Chez Josephine', 'Victor\'s Cafe', 'Al Bustan', 'Supreme Macaroni Co.',
            'BOS', 'Marti', 'Quartiere', 'McFeely\'s AmericanBistro', 'Pongsri Thai Restaurant', 'La Metairie',
            'Mortimer\'s', 'Bangkok House', 'Tripoli', 'Christine\'s', 'Contrapunto', 'Stella del Mare', 'Passports',
            'Pembroke Room', 'Villa Mosconi', 'Russell\'s American Grill', 'Cafe de Bruxelles', 'The Grange Hall',
            'Le Pistou', 'Gage & Tollner', 'Roumeli Taverna', 'TriBeCa Grill', 'Cucina & Co.', 'Mezzanine Restaurant',
            'Pierre\'s', 'India Pavilion', 'Take-Sushi', 'Avenue A', 'Mackinac Bar & Grill', 'Home', 'Lipstick Cafe',
            'Gino', 'Mamma Leone\'s', 'Asti', 'Carosello', 'Marylou\'s', 'Paradis Barcelona', 'La Chandelle',
            'Universal Grill', 'Mughlai', 'La Mediterranee', 'Grappino', 'Prix Fixe', 'Park Bistro', 'Patzo',
            'Adrienne', 'Hamburger Harry\'s', 'Chin Chin', 'Peacock Alley', 'Basta Pasta', 'Silk Road Palace',
            'Bangkok Cuisine', 'PARK AVENUE CAFE', 'Le Taxi', 'ARCADIA', 'Embers', 'Cornelia Street Cafe',
            'Brunetta\'s', 'Nanni\'s', 'Vincent\'s', 'Five Oaks', 'Dallas BBQ', 'Lanza Restaurant', 'Apple Restaurant',
            'Coconut Grill', 'The Ballroom', 'Cinquanta (aka 50)', 'Jerry\'s', 'Great American Health Bar',
            'Michael\'s', 'Harry Cipriani', 'LES CELEBRITES', 'Bill\'s Gay 90\'s', 'Mingala Burmese', 'Empire Szechuan',
            'Ipanema', 'Ponticello', 'Duane Park Cafe', 'Est Est Est', 'Harbour Lights', 'Pappardella', 'Baraonda',
            'Triplet\'s Roumanian', 'Dolcetto', 'Paola\'s', 'Piccolino', 'Sotto Cinque', 'Elaine\'s',
            'French Roast Cafe', 'Broome Street Bar', 'Rusty Staub\'s on Fifth', 'Trattoria Siciliana',
            'Istanbul Cuisine', 'Il Vigneto', 'Symphony Cafe', 'Itcho', 'Osso Buco', 'Hunan Garden', 'Soup Burg',
            'Supper Club', 'Chelsea Grill', 'MITALI EAST', 'Chelsea Central', 'Stellina', 'Perk\'s Fine Cuisine',
            'Raoul\'s on Varick', 'Tina\'s', 'Levana', 'Tommy Tang\'s', 'Elias Corner', 'Knickerbocker', 'El Faro',
            'Bubby\'s', 'Il Ponte', 'New Deal', 'Paris Commune', 'Mr. Tang', 'Cite', 'Beijing Duck House', 'Grove',
            'Thai Chef', 'Arturo\'s Pizzeria', 'The Water Club', 'RAINBOW ROOM', 'May We', 'Twigs', 'Landmark Tavern',
            'Lamarca', 'Sambuca', 'Metropolis Cafe', 'Cucina & Co.', 'Chicken Chef', 'Rose Cafe', 'Canton', 'Presto\'s',
            'Moreno', 'Chez Michallet', 'California Burrito Co.', 'Gingerty', 'Mayfair', 'Vinnie\'s Pizza',
            'Kitchen Club', 'Frontiere', 'La Petite Auberge', 'Pasta Lovers', 'Iso', 'John\'s Pizzeria', 'Corrado',
            'Serendipity 3', 'Grotta Azzurra', 'Jane Street Seafood Cafe', 'Silver Swan', 'Flying Fish', 'Sequoia',
            'Amsterdam\'s', 'Keewah Yen', 'Ernie\'s', 'Cafe Word of Mouth', 'Yaffa Cafe', 'Rocking Horse Mexican Cafe',
            'Lobster Box', 'Perretti Italian Cafe', 'Turkish Kitchen', 'Loui Loui', 'LE PERIGORD', 'Lola',
            'The Coach House', 'East', 'Ottomanelli\'s Cafe', 'Tevere 84', 'Brother Jimmy\'s BBQ', 'The Box Tree',
            'Ferrier', 'Cafe Un Deux Trois', 'Meriken', 'Bella Donna', 'Coming Or Going', 'Saranac', 'SHUN LEE PALACE',
            'Bello', 'La Barca', 'Andalousia (fka Lotfe\'s)', 'Taormina', 'Hi-Life Bar and Grill', 'Moran\'s',
            'Sakura of Japan', '7th Regiment Mess', 'Cucina Stagionale', 'Azzurro', 'L\'Ecole', 'Break for the Border',
            'Jade Palace', 'Follonico', 'Da Silvano', 'Szechuan Hunan Cottage', 'Steak Frites',
            'Tequila Sunrise(fka Tequila Willie\'s)', 'Capriccioso', 'Red Tulip', 'Sette MOMA', 'Bella Luna',
            'Three of Cups', 'Vernon\'s Jerk Paradise', 'T.S. Ma', 'Juanita\'s', 'Cafe Crocodile', 'Dok Suni',
            'Pellegrino', 'Fujiyama Mama', 'Siu Lam Kung', 'Docks Oyster Bar', 'Da Tommaso', 'Hosteria Fiorella',
            'Luma', 'Vico', 'UNION SQUARE CAFE', 'La Primavera', 'Barbetta', 'The Captain\'s Table', 'Boulevard',
            'Kaptain Banana', 'Scaletta', 'Canastel\'s', 'Oscar\'s', 'Vasata', 'Zinno', 'Country Club',
            'Totonno Pizzeria Napolitano', 'Sylvia\'s Restaurant', 'Carlyle Dining Room', 'Ambrosia Tavern',
            'Mme. Romaine de Lyon', 'Tiziano Trattoria', 'Cactus Cafe', 'IL NIDO', 'West End Gate', 'Au Troquet',
            'Allegria', 'Hop Shing', 'Manhattan Cafe', 'Sala Thai', 'Mandarin Court', 'Cupcake Cafe', 'Cafe Tabac',
            'Shaliga Thai Cuisine', 'Sukhothai West', 'L\'Auberge du Midi', '21 CLUB', 'Coastal', 'Seryna', 'ZARELA',
            'Jai Ya Thai', 'Mazzei', 'Park Side', 'Lattanzi Ristorante', 'Lion\'s Head', 'Ludlow Street Cafe',
            'Charlotte', 'The Nice Restaurant', 'Island Spice', 'Parma', 'Ruth\'s Chris Steak House', 'Diva',
            'La Bonne Soupe', 'West Broadway Restaurant', 'Karyatis', 'Hunters', 'Le Boeuf a la Mode', 'Bukhara',
            'Andiamo', 'Brazilian Pavilion', 'Pizzeria Uno', 'Hurricane Island', 'Little Poland', 'Bull & Bear',
            'Chez Jacqueline', 'Mueng Thai', 'Hong Fat', 'Pizzapiazza', 'Triangolo', 'Chef Ho\'s',
            'Manganaro\'s Hero-Boy', 'Cinema Diner', 'Briscola', 'La Luncheonette', 'Cleopatra\'s Needle',
            'Churchill\'s', 'Kabul Cafe', 'Cafe Metairie', 'Peruvian Restaurant', 'La Fusta', 'Time & Again Restaurant',
            'Pomodori', 'Merchants', 'Village Atelier', 'Le Biarritz', 'The Saloon', 'Passage To India',
            'Chelsea Commons', 'AUREOLE', 'Louie\'s Westside Cafe', 'Luke\'s Bar and Grill', 'L\'Auberge',
            'Oriental Garden', 'Angels', 'Charley O\'s', 'Garibaldi', 'RAO\'S', 'Lenge', 'O\'Neal\'s',
            'Gallagher\'s Steak House', 'Il Corallo Trattoria', 'Halcyon', 'Caffe Rafaella', 'Bombay Palace',
            'American Festival Cafe', 'Ellen\'s Stardust Diner', 'La Boite en Bois', 'S.P.Q.R.', 'Pig Heaven',
            'Four Winds', 'Caffe Bond', 'Malaga', 'Oriental Pearl', 'Market Cafe', 'SIGN OF THE DOVE', 'Samplings',
            'Orbit', 'Rose of India', 'Birdland', 'Madeo', 'Brighton Grill & Oyster Bar', 'Googies Italian Diner',
            'Demarchelier', 'Le Bistrot de Maxim\'s', 'Barocco', 'oh la la', 'Orson\'s', 'Sawadee Thai Cuisine',
            'Table d\'Hte', 'Meridiana', 'Edison Cafe', 'Ear Inn', 'Tout Va Bien', 'Rumpelmayer\'s',
            'Chefs & Cuisiniers Club', 'DANIEL (aka Restaurant Daniel)', 'Choshi', 'Ciccio & Tony\'s',
            'Friend of a Farmer', 'Cafe Loup', 'La Dolce Vita', 'Mangia e Bevi', 'Elio\'s', 'Shun Lee West',
            'Woo Lae Oak of Seoul', 'Arizona 206 and Cafe', 'Boca Chica', 'Olde Galway', 'TRIONFO', 'Ruppert\'s',
            'Tatany', 'Florence (fka Caffe Florence)', 'Grifone', 'Clarke\'s P.J.', 'MESA GRILL', 'Avanti',
            'Lusardi\'s', 'Pandit', 'Caffe Cefalu', 'Bistro 95', 'Ambassador Grill', 'Barking Dog Luncheonette',
            'Montebello', 'Bruno Ristorante', 'DA UMBERTO', 'Joe & Joe', 'Haveli', 'Isle of Capri', 'Penang Malaysia',
            'Katz\'s Deli', 'Bo Ky', 'Langan\'s', 'Blue Ribbon', 'Jean Lafitte', 'Dix et Sept',
            'Restaurant Two Two Two', 'Harry\'s Burritos', 'Christo\'s Steak House', 'Lofland\'s N.Y. Grill', 'Bayamo',
            'MANHATTAN OCEAN CLUB', 'Steamer\'s Landing', 'Sidewalker\'s', 'Road to Mandalay', 'The Shelby',
            'Col Legno', 'Nosmo King', 'La Topiaire', 'New Chao Chow', 'Lum Chin', 'The N.Y. Delicatessen', 'Kin Khao',
            'Anglers & Writers', 'La Taza De Oro', 'Bimini Twist', 'Cafe S.F.A.', 'Palm Too', 'CARNEGIE DELI',
            'Le Bilboquet', 'LESPINASSE', 'Seattle Bean Co.', 'Bloom\'s Delicatessen', 'Millenium Grill', 'Paolucci\'s',
            'Lello Ristorante', 'Ferdinando\'s Focacceria', 'Tre Pomodori', 'Marti Kebab', 'Mocca Hungarian', 'TERRACE',
            'Gramercy Watering Hole', 'China Grill', 'Afghan Kebab House', 'Delmonico\'s', 'Honeysuckle',
            'EJ\'s Luncheonette', 'Life Cafe', 'Khyber Pass', 'Aunt Sonia\'s', 'Colors', 'Two Boots', 'Vivolo',
            'Cafe Lalo', 'Patsy\'s', 'World Yacht', 'MARCH', 'Cho-Sen Garden', 'Piccolo Pomodoro', 'Lucky Cheng\'s',
            'Veniero\'s', 'Bali Burma', 'Borsalino', 'Sazerac House', 'Fresco Tortilla Grill', 'Rene Pujol',
            'U.S.A. Border Cafe', 'Live Bait', 'Sugar Reef', 'Lora', 'Sabrina Monet', 'Original California Taqueria',
            'Baci', 'Bice', 'Scalinatella', 'Rikyu', '1st Wok', 'Buona Sera', 'Cent\' Anni', 'CAFE DES ARTISTES',
            'Albuquerque Eats', 'Bistro du Nord', 'Mayrose', 'Nha Trang', 'Mee Noodle Shop', 'Le Pescadou', 'Chao Chow',
            'Le Madri', 'Arriba Arriba', 'Villa Berulia', 'Trattoria Alba', 'Il Monello', 'Sapporo East', 'Sardi\'s',
            'Tony\'s Di Napoli', 'Christ Cella', 'Bienvenue Restaurant', 'Sushi Zen', 'Gandhi', 'Metropolitan Cafe',
            'Zoe', 'Ruby\'s River Road', 'Barolo', 'Kiiroi Hana', 'Pipeline', 'Pasta Presto', 'Savoy', 'Coffee Shop',
            'Pier 25A', 'T.G.I. Friday\'s', 'Phoebe\'s', 'Spring Street Natural', 'La Maison Japonaise', 'Eva\'s Cafe',
            'Yellowfingers', '101', 'Zen Palate (93)', 'Remi', 'Dominick\'s Real Homestyle', 'Cottonwood Cafe',
            'Kiss Bar & Grill', 'Rancho Mexican Cafe', 'FOUR SEASONS', 'Patsy\'s Pizza', 'Edwardian Room', 'Indochine',
            'Zutto', 'SMITH & WOLLENSKY', 'Shun Lee Cafe', 'Teachers Too', '2nd Ave. Brasserie', 'Caliente Cab Co.',
            'Tortilla Flats', 'Sant Ambroeus', 'Manhattan Chili Co.', 'Pudgie\'s', '44 Southwest', 'Les Friandises',
            'Galil', 'Mi Cocina', 'Taste of Hong Kong', 'Les Routiers', 'Medici 56', 'La Boulangere', 'Caribe',
            'Fifty-Seven', 'Sfuzzi', 'Mr. Souvlaki', 'Patrissy\'s', 'JO JO', 'Le Veau d\'Or', 'Tacomadre',
            'Mike\'s Bar & Grill', 'Ollie\'s', 'Patisserie J. Lanciani', 'Manila Gardens', 'The Levee',
            'Hwa Yuan Szechuan Inn', 'Nusantara', 'Ecco L\'Italia', 'Coco Pazzo', 'Bistro 36', 'Le Max',
            'Brother\'s Bar-B-Q', 'Boathouse Cafe', 'Great Jones Cafe', 'La Mangeoire', 'SECOND AVENUE DELI',
            'GOTHAM BAR & GRILL', 'Lobby Lounge', 'La Focaccia', 'Portico', 'Bertha\'s', 'Gianni\'s', 'Chumley\'s',
            'Taste of Tokyo', 'Main Street', 'Regency', 'Corner Bistro', 'Parioli Romanissimo', 'Cafe Trevi',
            'Siam Inn', 'Sweet Basil\'s', 'La Collina', 'Cowgirl Hall of Fame', 'La Goulue', 'Cuisine de Saigon',
            'Fraunces Tavern Restaurant', 'Zip City Brewery', 'Tibetan Restaurant', 'Pen & Pencil Restaurant',
            'Papa Bear', 'Vong & Kwong', 'Cafe Botanica', 'Chelsea Trattoria', 'Mezzaluna', 'Billy\'s', 'Emily\'s',
            'Florent', 'Bar Pitti', 'Szechuan Kitchen', 'La Palette', 'Pasticcio', 'Jackson Hole', 'Pigalle',
            'Cafe de Paris', 'Vespa', 'Rachel\'s', 'Tre Giganti', 'Casalone', 'Gray\'s Papaya', 'Viand', 'DeGrezia',
            'Provence', 'Ecco', 'Pierre au Tunnel', 'Honmura An', 'Shinbashi-an', 'Mickey Mantle\'s']
