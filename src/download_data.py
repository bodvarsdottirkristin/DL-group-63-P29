from datetime import datetime, timedelta
import os
import requests

def download_ais_range(start_date: datetime,
                       end_date: datetime,
                       output_dir: str,
                       base_url: str = "http://aisdata.ais.dk/aisdk-{date}.zip"):
    """
    Download AIS .zip files for dates in [start_date, end_date).

    Parameters
    ----------
    start_date : datetime
        First date to download (inclusive).
    end_date : datetime
        Last date to download (exclusive).
    output_dir : str
        Directory to save downloaded .zip files.
    base_url : str
        URL template; must contain "{date}" placeholder formatted as YYYY-MM-DD.
    """

    os.makedirs(output_dir, exist_ok=True)
    current = start_date

    print(f"Downloading AIS data to: {output_dir}")
    print(f"Date range: {start_date.date()} → {end_date.date()} (exclusive)\n")

    while current < end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = base_url.format(date=date_str)
        out_path = os.path.join(output_dir, f"aisdk-{date_str}.zip")

        print(f"→ {date_str}: {url}")

        try:
            r = requests.get(url, stream=True, timeout=30)

            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"   Saved: {out_path}")
            else:
                print(f"   File not found (HTTP {r.status_code})")

        except requests.exceptions.RequestException as e:
            print(f"   Error downloading: {e}")

        current += timedelta(days=1)

    print("\nDone!")
