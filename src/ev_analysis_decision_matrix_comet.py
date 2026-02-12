import pandas as pd
import numpy as np
from functions.mcda_functions import *
import json
import os

def fill_decision_matrix():
    """
    Wypełnia macierz decyzyjną dla lokalizacji stacji EV danymi z API
    """
    
    # Twoje 9 punktów
    points = [
        (53.432733, 14.541102),  # Punkt 1 - Dobry
        (53.421491, 14.535882),  # Punkt 2 - Dobry  
        (53.441069, 14.513299),  # Punkt 3 - Dobry
        (53.418425, 14.564761),  # Punkt 4 - Średni
        (53.441944, 14.575778),  # Punkt 5 - Średni
        (53.404396, 14.562735),  # Punkt 6 - Średni
        (53.428648, 14.559362),  # Punkt 7 - Słaby
        (53.431307, 14.563053),  # Punkt 8 - Słaby
        (53.409585, 14.534809),  # Punkt 9 - Słaby
    ]
    
    # Nazwy punktów
    point_names = [f"Punkt_{i+1}" for i in range(len(points))]
    
    # Kategorie
    categories = ["Dobre", "Dobre", "Dobre", "Średnie", "Średnie", "Średnie", "Słabe", "Słabe", "Słabe"]
    
    # Definicja kryteriów zgodnie z Twoją listą
    criteria_definitions = [
        {
            "name": "Odległość od parkingów",
            "method": "distance",
            "api_params": ("amenity", "parking"),
            "description": "Odległość w metrach do najbliższego parkingu"
        },
        {
            "name": "Odległość od głównych dróg", 
            "method": "distance",
            "api_params": ("highway", "primary"),
            "description": "Odległość w metrach do najbliższej głównej drogi"
        },
        {
            "name": "Odległość do istniejących stacji",
            "method": "distance", 
            "api_params": ("amenity", "charging_station"),
            "description": "Odległość w metrach do najbliższej stacji ładowania"
        },
        {
            "name": "Ilość sklepów w okolicy",
            "method": "count",
            "api_params": ("shop", "supermarket"),
            "description": "Liczba sklepów w promieniu 500m"
        },
        {
            "name": "Ilość restauracji w okolicy", 
            "method": "count",
            "api_params": ("amenity", "restaurant"),
            "description": "Liczba restauracji w promieniu 500m"
        },
        {
            "name": "Gęstość zaludnienia w okolicy",
            "method": "count",
            "api_params": ("building", "residential"), 
            "description": "Liczba budynków mieszkalnych w promieniu 500m"
        }
    ]
    
    print("🚀 Rozpoczynam pobieranie danych z API...")
    print(f"Liczba punktów do analizy: {len(points)}")
    print(f"Liczba kryteriów: {len(criteria_definitions)}")
    
    # Pobieranie danych dla każdego punktu
    results_matrix = []
    
    for i, (lat, lon) in enumerate(points):
        print(f"\n📍 Przetwarzam punkt {i+1}/9: ({lat}, {lon})")
        
        point_data = []
        for j, criterion in enumerate(criteria_definitions):
            print(f"  ⏳ Kryterium {j+1}/{len(criteria_definitions)}: {criterion['name']}")
            
            try:
                tag_key, tag_value = criterion["api_params"]
                
                if criterion["method"] == "distance":
                    # Szukamy najbliższego obiektu
                    value = find_nearest_poi(lat, lon, tag_key, tag_value, radius=500, delay=1.0)
                    print(f"     ✅ Odległość: {value:.1f}m")
                    
                elif criterion["method"] == "count":
                    # Liczymy obiekty w promieniu
                    value = count(lat, lon, tag_key, tag_value, radius=500, delay=1.0)
                    print(f"     ✅ Liczba: {value}")
                
                point_data.append(value)
                
            except Exception as e:
                print(f"     ❌ Błąd: {e}")
                point_data.append(0.0)
        
        results_matrix.append(point_data)
        print(f"  ✅ Punkt {i+1} zakończony")
    
    # Tworzenie DataFrame
    print("\n📊 Tworzę macierz decyzyjną...")
    
    criteria_columns = [
        "Odległość_od_parkingów_m",
        "Odległość_od_głównych_dróg_m", 
        "Odległość_do_istniejących_stacji_m",
        "Ilość_sklepów_szt",
        "Ilość_restauracji_szt",
        "Gęstość_zaludnienia_os_km2"
    ]
    
    df = pd.DataFrame({
        'ID': range(1, len(points) + 1),
        'Współrzędne': [f"{lat}, {lon}" for lat, lon in points],
        'Lat': [lat for lat, lon in points],
        'Lon': [lon for lat, lon in points],
        'Kategoria': categories
    })
    
    # Dodawanie kolumn z kryteriami
    for i, col_name in enumerate(criteria_columns):
        df[col_name] = [row[i] for row in results_matrix]
    
    # Dodanie kolumn do dalszej analizy
    df['Wynik_całkowity'] = ""
    df['Ranking'] = ""
    
    # Zapisanie do plików
    output_csv = "macierz_decyzyjna_wypelniona.csv"
    output_json = "macierz_decyzyjna_wypelniona.json"
    
    # CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"💾 Zapisano CSV: {output_csv}")
    
    # JSON dla dodatkowych informacji
    results_dict = {
        "metadata": {
            "czas_analizy": pd.Timestamp.now().isoformat(),
            "liczba_punktow": len(points),
            "liczba_kryteriow": len(criteria_definitions),
            "promien_analizy_m": 500
        },
        "kryteria": [
            {
                "id": i+1,
                "nazwa": crit["name"],
                "opis": crit["description"],
                "metoda": crit["method"],
                "api_params": crit["api_params"]
            }
            for i, crit in enumerate(criteria_definitions)
        ],
        "punkty": points,
        "wyniki": results_matrix
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 Zapisano JSON: {output_json}")
    
    # Wyświetlenie podsumowania
    print("\n📈 PODSUMOWANIE WYNIKÓW:")
    print("=" * 80)
    
    for i, (point_name, category) in enumerate(zip(point_names, categories)):
        print(f"\n{point_name} ({category}):")
        for j, (criterion, value) in enumerate(zip(criteria_definitions, results_matrix[i])):
            if criterion["method"] == "distance":
                print(f"  • {criterion['name']}: {value:.1f} m")
            else:
                print(f"  • {criterion['name']}: {int(value)} szt.")
    
    print(f"\n✅ Analiza zakończona! Sprawdź pliki:")
    print(f"   📄 {output_csv}")
    print(f"   📄 {output_json}")
    
    return df, results_dict

if __name__ == "__main__":
    # Upewnij się, że katalogi istnieją
    os.makedirs("output", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Uruchom analizę
    df, results = fill_decision_matrix()
    
    # Wyświetl pierwszych kilka wierszy
    print("\n🔍 Pierwsze 3 wiersze macierzy:")
    print(df.head(3).to_string())
