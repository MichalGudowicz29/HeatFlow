<p align="center">
  <img src="assets/banner.png" alt="HeatFlow – mapa, która pokazuje, gdzie najlepiej postawić to, co planujesz w mieście" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="Licencja MIT" src="https://img.shields.io/badge/licencja-MIT-5b6b6d"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-1f4e8c?logo=python&logoColor=white">
  <img alt="Dane: OpenStreetMap" src="https://img.shields.io/badge/dane-OpenStreetMap-7ebc6f?logo=openstreetmap&logoColor=white">
  <img alt="Metody: SPOTIS, TOPSIS, VIKOR" src="https://img.shields.io/badge/metody-SPOTIS%20%C2%B7%20TOPSIS%20%C2%B7%20VIKOR-d7191c">
  <img alt="Testy: 45 przechodzi" src="https://img.shields.io/badge/testy-45%20%E2%9C%93-1d6b45">
</p>

<p align="center">
  <img src="assets/award-forum-teleinformatyki.svg" alt="Wyróżnienie w Konkursie Młodych Mistrzów, XXXI Forum Teleinformatyki 2025" width="400">
  &nbsp;
  <img src="assets/publication-kes-2025.svg" alt="Publikacja naukowa KES 2025" width="400">
</p>

<p align="center">
  <b>Gdzie postawić stację roweru miejskiego? Gdzie nową ładowarkę? Gdzie zbudować plac zabaw?</b><br>
  HeatFlow odpowiada na takie pytania mapą: im cieplejszy kolor, tym lepsze miejsce.
</p>

<p align="center">
  <img src="assets/screenshots/hero-szczecin.png" alt="Mapa całego Szczecina: najlepsze miejsca na nowe stacje roweru miejskiego" width="100%">
</p>

## Jak to działa

Wybierasz obszar i ustalasz, co jest ważne. HeatFlow ocenia miejsca na mapie i wskazuje 10 najlepszych lokalizacji.

**Plac zabaw na Pogodnie.** Zwiększamy wagę bliskości parków z 15% do 54%. HeatFlow od razu przelicza mapę i wskazuje nowe najlepsze miejsce.

<p align="center">
  <img src="assets/screenshots/zmiana-wagi.gif" alt="Po zwiększeniu wagi kryterium Parki miejsca bliżej parków na Pogodnie zyskują na mapie oceny." width="100%">
</p>

[Otwórz animację](assets/screenshots/zmiana-wagi.gif) · [Zobacz instrukcję krok po kroku](docs/instrukcja.md)

## Przykłady

<table>
  <tr>
    <td width="50%"><img src="assets/screenshots/przyklad-ev-miasto.png" alt="Ładowarki samochodowe w całym Szczecinie"></td>
    <td width="50%"><img src="assets/screenshots/przyklad-osiedle.png" alt="Nowy plac zabaw na osiedlu Pogodno"></td>
  </tr>
  <tr>
    <td><b>Ładowarki samochodowe w całym mieście</b><br>Daleko od istniejących ładowarek, blisko głównych dróg i parkingów.</td>
    <td><b>Plac zabaw na jednym osiedlu</b><br>Można zbadać też mały fragment miasta, z dokładnością do ulicy.</td>
  </tr>
</table>

Każdy przykład to gotowy scenariusz z [`examples/`](examples/), który otwiera się jednym kliknięciem w aplikacji.

## Co jeszcze potrafi

- **Odległość pieszo**, liczona po chodnikach i ścieżkach, albo w linii prostej.
- **Wyjaśnienie wyniku**: po kliknięciu w mapę widać, które kryterium podniosło, a które obniżyło ocenę.
- **Pomijanie terenów** takich jak woda, lasy czy cmentarze.
- **Eksport** do GeoJSON, CSV i PNG oraz zapis całej analizy do pliku, żeby wrócić do niej później.
- **Działa lokalnie**: dane Szczecina z OpenStreetMap są na Twoim komputerze, analiza całego miasta trwa kilka–kilkanaście sekund.

<p align="center">
  <img src="assets/screenshots/przyklad-punkt.png" alt="Wyjaśnienie, dlaczego wybrany punkt zajął pierwsze miejsce" width="100%">
</p>

## Dokumentacja

- [Instrukcja obsługi krok po kroku](docs/instrukcja.md), ze zrzutami ekranu
- [Metodyka i szczegóły techniczne](docs/metodyka.md): jak liczone są odległości, wagi i oceny

## Autorzy

<img src="assets/data-engine.png" alt="Koło naukowe Data Engine" width="180" align="right">

**Michał Gudowicz**, **Szymon Śniegowski**. Opieka naukowa: **dr hab. inż. Wojciech Sałabun**.
Koło naukowe **Data Engine**, Zachodniopomorski Uniwersytet Technologiczny w Szczecinie.

- Publikacja: Gudowicz M., Śniegowski S., Sałabun W., *An MCDA-based decision-making framework for
  evaluating virtual bike-sharing station locations with spatial data*, KES 2025.
- Wyróżnienie Komitetu Programowo-Organizacyjnego w Konkursie Młodych Mistrzów, Forum
  Teleinformatyki 2025.

## Wypróbuj u siebie

Potrzebny jest [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python zainstaluje się sam).

```bash
git clone https://github.com/MichalGudowicz29/HeatFlow.git
cd HeatFlow
uv sync --extra importer
uv run heatflow import-region   # jednorazowo: dane Szczecina z OpenStreetMap (ok. 100 MB, ok. 1 min)
uv run heatflow serve --port 5050   # otwórz http://127.0.0.1:5050
```

Gotowe przykłady są w pierwszym kroku aplikacji (przyciski **Przykład: …**). Własne analizy zapisuje się i otwiera przyciskami **Zapisz** i **Wczytaj**.

<sub>Kod na licencji [MIT](LICENSE). Dane mapy © współtwórcy OpenStreetMap, licencja ODbL.</sub>
