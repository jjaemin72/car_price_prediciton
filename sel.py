from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from dataclasses import dataclass, asdict
from typing import List
import csv
import logging
from bs4 import BeautifulSoup

@dataclass
class Car:
    link: str
    full_name: str
    description: str
    year: int
    mileage: str
    price: int

class CarScraper:
    def __init__(self, car_make: str) -> None:
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.11 (KHTML, Like Gecko) "
            "Chrome/23.0.1271.64 Safari/537.11",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
            "Accept-Encoding": "none",
            "Accept-Language": "en-US, en;q=0.8",
            "Connection": "keep-alive",
        }
        self.car_make = car_make
        self.website = f"https://www.cargurus.com/Cars/l-Used-{car_make}-m25#resultsPage=1"

        # Set up Chrome options
        options = Options()
        options.add_argument("--headless")  # Run Chrome in headless mode
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Path to chromedriver
        self.driver = webdriver.Chrome(service=Service('/usr/local/bin/chromedriver'), options=options)

    def load_page(self, page_number: int):
        url = f"{self.website}?page={page_number}"
        self.driver.get(url)

    def get_page_source(self) -> str:
        return self.driver.page_source

    def extract_cars_from_page(self, soup: BeautifulSoup) -> List[Car]:
        offers_table = soup.find('div', id='cargurus-listing-search')
        
        if offers_table is None:
            print("Failed to find the offers table on the page.")
            logging.error("Failed to find the offers table on the page.")
            return []
        
        print("Found offers table.")
        
        # Navigate through nested divs to find car entries
        nested_container = offers_table.find('div', class_='pecvNo')
        if nested_container is None:
            print("Failed to find the nested container.")
            logging.error("Failed to find the nested container.")
            return []
        
        car_entries_container = nested_container.find('div', class_='Km1Vfz')
        if car_entries_container is None:
            print("Failed to find the car entries container.")
            logging.error("Failed to find the car entries container.")
            return []
        
        car_entries = car_entries_container.find_all('div', class_='pazLTN')
        print(f"Found {len(car_entries)} car entries.")  # Debugging line

        list_of_cars = []
        for car in car_entries:
            try:
                # Safely extract car details with default values if elements are not found
                link = car.find('a', href=True).get('href', 'N/A')
                full_name = car.find('h4').get('title', 'N/A')
                
                # Safely split full_name to get the year and the model
                split_name = full_name.split()
                year = int(split_name[0]) if split_name[0].isdigit() else 0
                model_name = " ".join(split_name[1:])

                mileage = car.find('p', class_='us1dS iK3Zj _erOpv kIL3VY', attrs={'data-testid': 'srp-tile-mileage'})
                mileage_text = mileage.text.strip() if mileage else 'N/A'

                price = car.find('h4', class_='us1dS i5dPf SOf0Fe')
                price_text = price.text.strip().replace("$", "").replace(",", "") if price else '0'
                description = car.find('p', class_='us1dS iK3Zj _erOpv kIL3VY', attrs={'data-testid': 'seo-srp-tile-engine-display-name'})
                description_text = description.text.strip() if description else 'N/A'


                car_data = Car(
                    link="https://www.cargurus.com" + link,
                    full_name=model_name,
                    description=description_text,
                    year=year,
                    mileage=mileage_text,
                    price=int(price_text)
                )
                list_of_cars.append(car_data)
                print(f"Extracted car: {car_data}")  # Debugging line
                logging.info(f"Extracted car: {car_data}")
            except Exception as e:
                logging.error(f"Failed to gather car. Msg: {e}")
                print(f"Failed to gather car. Msg: {e}")
        return list_of_cars

    def quit_driver(self):
        if self.driver:
            self.driver.quit()

def write_to_csv(cars: List[Car]) -> None:
    try:
        with open("cars_first_page.csv", mode="w", newline='') as f:
            fieldnames = [
                "link",
                "full_name",
                "description",
                "year",
                "mileage",
                "price",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for car in cars:
                writer.writerow(asdict(car))
    except IOError as e:
        print(f"Failed to write to CSV file. Msg: {e}")

def scrape_cargurus():
    make = "Ferrari"  # Adjust this as needed
    scraper = CarScraper(make)
    
    try:
        # Load the first page
        scraper.load_page(1)
        
        # Extract cars from the first page
        cars = scraper.extract_cars_from_page(BeautifulSoup(scraper.get_page_source(), "html.parser"))
        print(f"Number of cars found on the first page: {len(cars)}")
        logging.info(f"Number of cars found on the first page: {len(cars)}")
        
        # Save to a CSV file
        write_to_csv(cars)
        print("Cars saved to cars_first_page.csv")
        logging.info("Cars saved to cars_first_page.csv")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    finally:
        scraper.quit_driver()

if __name__ == '__main__':
    scrape_cargurus()
