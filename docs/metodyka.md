# HeatFlow. Przebieg i metodyka

## Przebieg analizy

Interfejs prowadzi przez sześć etapów opisanych w projekcie.

1. **Problem decyzyjny.** Nazwa i cel analizy.
2. **Kryteria i wagi.** Kryteria z katalogu OpenStreetMap (76 warstw w 9 grupach), dowolny własny tag
   OSM albo wgrany plik GeoJSON/CSV. Dla każdego kryterium wybierasz:
   - pomiar: odległość do najbliższego obiektu (w linii prostej albo pieszo po sieci dróg) lub liczbę
     obiektów w promieniu α,
   - kierunek preferencji: więcej/dalej = lepiej albo mniej/bliżej = lepiej.

   Kryterium może obejmować wszystkie obiekty danego rodzaju (np. wszystkie uczelnie) albo tylko
   wskazane z nazwy (np. kampusy ZUT). Obiekty o tej samej nazwie wybiera się razem.

   Wagi: ręczne, RANCOM, obiektywne z danych (entropia, CRITIC, odchylenie standardowe) albo równe.
   RANCOM (Więckowski i in., 2023) prowadzi serię pytań, tak jak `RANCOM(object_names=…)`
   w pymcdm. Dla każdej pary kryteriów wybierasz: pierwsze ważniejsze (1), równie ważne (½) albo
   drugie ważniejsze (0). Z odpowiedzi powstaje macierz porównań MAC, a wagi wylicza pymcdm
   (`RANCOM(matrix=MAC)`, waga = SCW / ΣSCW). Interfejs pokazuje macierz, sumy SCW i liczbę
   niespójnych trójek odpowiedzi (diagnostyka – RANCOM wylicza wagi także przy sprzecznych
   odpowiedziach, ale ich nie poprawia). Po dodaniu kryterium pytania dotyczą tylko nowych par.
   Ważność można też podać rankingiem albo punktacją 0–100. Punktacja jest zamieniana na ranking
   z remisami i liczona przez `RANCOM(ranking=…)`, bo `RANCOM(scoring=…)` w pymcdm 1.4.0 przy
   powtórzonych wartościach przestawia pozycje.
3. **Obszar analizy.** Całe miasto, granica osiedla z wyszukiwarki albo prostokąt lub wielokąt
   narysowany na mapie. Obszar jest przycinany do granicy miasta.
4. **Punkty analizy.** Regularna siatka w metrach przycięta do obszaru. Rozstaw ręczny albo
   automatyczny, dobrany do wielkości obszaru (ok. 3000 punktów; to dobór według powierzchni, nie
   według gęstości danych). Wody (domyślnie, także w CLI i scenariuszu bez pola `exclude`), lasy
   i cmentarze można wyłączyć z oceny: punkty na tych terenach nie są oceniane, nie wpływają
   na skalę kolorów ani granice SPOTIS, a na mapie są szare.
5. **Analiza wielokryterialna.** SPOTIS (domyślnie), TOPSIS lub VIKOR.
6. **Mapa preferencji.** Interaktywna warstwa na mapie z wyborem palety, krycia i wygładzania.
   Dostępne są też:
   - histogram ocen w legendzie,
   - 10 najlepszych lokalizacji,
   - wartości kryteriów w najbliższym ocenionym węźle siatki (kliknięcie w mapę); straty
     poszczególnych kryteriów dla SPOTIS,
   - podgląd obiektów źródłowych,
   - eksport do GeoJSON, CSV i PNG.

Zmiana wag, metody lub kierunku kryterium przelicza mapę od razu, bez ponownego pobierania danych.
Zmiana kryteriów, obszaru lub siatki wymaga ponownego wygenerowania. Scenariusz (cała konfiguracja)
zapisuje się do pliku JSON i wczytuje z powrotem. Ten sam format przyjmuje `heatflow run`.

## Metodyka

- **Dane.** Domyślnie lokalny region: SQLite z indeksem przestrzennym R-tree i indeksem tagów.
  Multipoligony są składane z ról `outer`/`inner` (z otworami), a graf pieszy jest zapisany jako
  tablice. Bufor 6 km poza granicą miasta sprawia, że odległości przy granicy nie są zawyżone.
  Bez regionu dane pobiera Overpass API: jedno zapytanie na kryterium dla obszaru z buforem,
  cache na dysku kluczowany treścią zapytania. Błąd pobierania zatrzymuje analizę z komunikatem
  i nigdy nie jest zastępowany wartością domyślną.
- **Geometria.** Lokalna projekcja równoodległościowa wokół środka obszaru z promieniami krzywizny
  elipsoidy WGS84. Błąd odległości względem geodezji rośnie z oddaleniem od środka: dla Szczecina
  z buforem 6 km wynosi maks. ok. 0,4% (ok. 0,25% w granicach miasta), czyli kilka metrów na
  1,5 km. Odległość liczona jest do geometrii obiektu: punktu, linii albo poligonu (wewnątrz
  poligonu = 0), a nie do jego środka.
- **Odległość piesza.** Graf pieszy z OSM: drogi i ścieżki bez autostrad, bez `foot=no` i bez
  `access=no/private`, chyba że `foot=yes/designated/permissive` (tag `foot` ma pierwszeństwo).
  Wieloźródłowa Dijkstra liczy od obiektów docelowych. Punkt siatki i obiekt dociągane są do
  najbliższych **odcinków** sieci (rzut prostopadły; odcinki najwyżej 10 m dalsze od najbliższego).
  Odcinek dociągnięcia nigdy nie przecina wody, torów kolejowych ani autostrady, a ogrodzeń
  i murów unika: przez ogrodzenie prowadzi tylko wtedy, gdy w pobliżu nie ma innego dojścia
  (OSM rzadko ma wszystkie bramy). Odcięte fragmenty sieci (< 50 węzłów) nie służą do
  dociągania. Gdy w promieniu 500 m nie ma dojścia, punkt dostaje wartość graniczną, a ostrzeżenie podaje ich liczbę osobno (przy kryterium „dalej = lepiej” taki punkt wypada najlepiej, więc warto takie tereny wyłączyć z oceny). Gdy punkt i obiekt leżą
  przy tym samym odcinku, odległość liczona jest bezpośrednio wzdłuż niego. Wynik nigdy nie jest
  mniejszy od odległości w linii prostej. Punkty bez trasy krótszej niż limit dostają wartość
  graniczną z osobnym ostrzeżeniem. Dostępna dla obszarów do 400 km². Bramy i wejścia
  do budynków nie są modelowane, a dojście od punktu do sieci liczone jest w linii prostej.
  Porównanie z niezależnym grafem (networkx, długości geodezyjne) daje medianę różnic kilku metrów.
- **Obcięcie odległości.** Odległości powyżej limitu (domyślnie 1,5 km, można ustawić osobno dla
  każdego kryterium) przyjmują wartość limitu („≥ 1500 m”) i są wtedy równorzędne. Limit jest
  częścią modelu: zmienia granice SPOTIS i może zmienić ranking (w przykładzie EV podniesienie
  z 1,5 do 3 km wymienia 3 z 10 najlepszych lokalizacji). Liczba punktów na limicie jest podawana
  w ostrzeżeniu. Obiekty z bufora poza granicą miasta (do 6 km) są uwzględniane zarówno
  w odległościach, jak i w licznościach.
- **SPOTIS.** Granice kryteriów to zakres wartości w ocenianych punktach, więc ocena jest
  względna: zmiana obszaru, rozstawu siatki albo limitu zmienia punkt odniesienia (odporność
  SPOTIS na odwrócenie rankingu obowiązuje przy stałych granicach). Wynik to ważona odległość od punktu
  idealnego (0 = ideał) i rozkłada się na straty poszczególnych kryteriów pokazywane w punkcie.
- **Kryteria stałe** (np. brak obiektów w zasięgu) nie różnicują lokalizacji. Są wykluczane
  z oceny, a wagi pozostałych normalizowane, z ostrzeżeniem w interfejsie.
- **Wagi obiektywne** mierzą zróżnicowanie kryteriów w wybranym obszarze i na wybranej siatce.
  Sąsiednie punkty są ze sobą skorelowane, więc przed publikacją warto sprawdzić stabilność wag
  i najlepszych lokalizacji przy innym rozstawie siatki. Wagi liczone są ze stopni spełnienia
  kryteriów: normalizacja min–max, a dla kryteriów kosztowych odwrócenie (1 = najlepsza wartość
  w obszarze) – CRITIC mierzy konflikt między stopniami spełnienia (Diakoulaki i in., 1995).
  Entropia daje większą wagę kryteriom o bardziej nierównomiernym rozkładzie, CRITIC – kryteriom
  zróżnicowanym i słabo albo przeciwnie skorelowanym z pozostałymi, odchylenie standardowe –
  proporcjonalnie do rozrzutu. Wagi obiektywne liczone są tylko z kryteriów, które różnicują
  obszar. Entropia stosuje konwencję 0·ln 0 = 0 (implementacja w pymcdm 1.4.0 zeruje entropię
  kolumny z choćby jednym zerem, co dla danych o liczności obiektów jest błędne).
- **Mapa.** Raster liczony piksel po pikselu w odwzorowaniu Web Mercator i przycięty do granicy
  obszaru (z otworami). Wygładzanie to interpolacja dwuliniowa między węzłami (bez przestrzeleń
  poza zakres ocen) i ma charakter wizualny. Ocena dotyczy węzłów siatki, a kliknięcie pokazuje
  najbliższy oceniony punkt.
- **Skala kolorów.** „Wartość oceny” pokazuje wielkość różnic: dla SPOTIS i VIKOR niżej = lepiej,
  dla TOPSIS wyżej = lepiej. „Pozycja w rankingu” (percentyl) pozwala porównywać metody, ale ukrywa
  wielkość różnic. Eksport zawsze zawiera surowy wynik i jego kierunek.
- **Liczność w promieniu α** zlicza obiekty, których punkt reprezentatywny (`point_on_surface`)
  leży w kole o promieniu α. Duże obiekty, które tylko częściowo wchodzą w koło, nie są liczone:
  dla parków przy α = 1 km daje to ok. 30% mniej obiektów niż liczenie „dowolną częścią
  w promieniu”, dla szkół i parkingów ok. 6–7%. Bez lokalnego regionu (Overpass `out center`)
  punktem obiektu jest środek obwiedni, który dla obiektów wklęsłych może leżeć poza nimi.
- **Odtwarzalność.** Wynik i eksport GeoJSON (interfejs i CLI) zawierają datę stanu bazy OSM,
  źródło danych i wyłączone tereny. CSV zawiera same dane punktów. Import regionu podmienia dane
  w całości naraz, a analiza do końca korzysta z wersji danych, z którą wystartowała.

## Dane i serwery zewnętrzne

- **OpenStreetMap.** Lokalny wyciąg z [Geofabrik](https://download.geofabrik.de/europe/poland.html),
  a bez regionu publiczne serwery Overpass z ponawianiem przy przeciążeniu (własny serwer: zmienna
  `HEATFLOW_OVERPASS_URL`).
- **Wyszukiwarka** przeszukuje lokalnie osiedla, ulice i obiekty miasta. Bez regionu korzysta
  z Nominatim (wyszukiwanie po Enterze, maks. 1 zapytanie na sekundę), zgodnie z
  [zasadami użycia](https://operations.osmfoundation.org/policies/nominatim/).
- **Podkład mapy** to kafelki OpenStreetMap (online).
- **Własne dane** w GeoJSON lub CSV (WGS84). Są zapisywane lokalnie w `.heatflow/datasets/`.

Dane © współtwórcy OpenStreetMap, licencja ODbL.

## Struktura

```
heatflow/   pakiet: geo (siatka), region (lokalne dane OSM), osm (Overpass), features, criteria,
            network (odl. piesza), weights, mcda, analysis, server (Flask API), cli
web/        interfejs (Leaflet, bez kroku budowania)
data/       katalog warstw OSM (catalog.json)
examples/   scenariusze (stacje ładowania EV – Szczecin)
tests/      testy na danych syntetycznych
```
