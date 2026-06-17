"""
Global Flood Monitoring (GFM) REST API client.

This module provides functionality to authenticate and download GFM flood data
from the GFM REST API (https://api.gfm.eodc.eu/v2/).
"""

import os
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

import requests
from shapely.geometry import box, Polygon, mapping

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GFMRestClient:
    """Client for GFM REST API authentication and data download."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: str = "https://api.gfm.eodc.eu/v2",
    ):
        """
        Initialize GFM REST API client.

        Args:
            username: GFM API username (or set GFM_API_USERNAME env var)
            password: GFM API password (or set GFM_API_PASSWORD env var)
            base_url: Base URL for GFM API (default: https://api.gfm.eodc.eu/v2)
        """
        self.base_url = base_url.rstrip("/")

        # Get credentials from params or environment
        self.username = username or os.getenv("GFM_API_USERNAME")
        self.password = password or os.getenv("GFM_API_PASSWORD")

        if not self.username or not self.password:
            raise ValueError(
                "Username and password required. Provide via parameters or set "
                "GFM_API_USERNAME and GFM_API_PASSWORD environment variables."
            )

        self.token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.client_id: Optional[str] = None
        self.session = requests.Session()

    def login(self) -> Dict:
        """
        Authenticate with GFM API and get bearer token.

        Returns:
            Login response containing token and user info

        Raises:
            requests.HTTPError: If authentication fails
        """
        url = f"{self.base_url}/auth/login"
        payload = {"email": self.username, "password": self.password}

        logger.info(f"Authenticating to GFM API as {self.username}")

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            self.token = data.get("access_token")
            self.client_id = data.get("client_id")
            # Use client_id as user_id (this works for the API)
            self.user_id = self.client_id

            if not self.token:
                raise ValueError("No access token in login response")

            # Set authorization header for future requests
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

            logger.info("Successfully authenticated to GFM API")
            return data

        except requests.HTTPError as e:
            logger.error(f"Authentication failed: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def create_aoi(
        self,
        name: str,
        geometry: Union[Polygon, dict, List[float]],
        description: Optional[str] = None,
    ) -> Dict:
        """
        Create an Area of Interest (AOI) on the GFM API.

        Args:
            name: Name for the AOI
            geometry: AOI geometry as Shapely Polygon, GeoJSON dict, or bbox [west, south, east, north]
            description: Optional description

        Returns:
            API response with AOI details including aoi_id

        Raises:
            requests.HTTPError: If AOI creation fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        url = f"{self.base_url}/aoi/create"

        # Convert geometry to GeoJSON if needed
        if isinstance(geometry, list):
            # Assume it's a bbox [west, south, east, north]
            geometry = box(*geometry)

        if isinstance(geometry, Polygon):
            geojson = mapping(geometry)
        else:
            geojson = geometry

        payload = {
            "aoi_name": name,  # API expects 'aoi_name' not 'name'
            "geoJSON": geojson,  # API expects 'geoJSON' not 'geometry'
            "client_id": self.client_id,
            "access_token": self.token,
            "user_id": self.user_id
        }

        if description:
            payload["description"] = description

        logger.info(f"Creating AOI: {name}")
        logger.debug(f"Payload: {payload}")

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            logger.info(f"AOI created successfully with ID: {data.get('aoi_id')}")
            return data

        except requests.HTTPError as e:
            logger.error(f"Failed to create AOI: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def list_aois(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        List all AOIs for a specific user.

        Args:
            user_id: User ID (defaults to self.user_id from login)

        Returns:
            List of AOI dictionaries

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        # Use provided user_id or fallback to instance user_id
        uid = user_id or self.user_id

        if not uid:
            raise ValueError("user_id required. Either provide it or ensure login() was called first.")

        # Try the /aoi/user/{user_id} endpoint from the tutorial
        url = f"{self.base_url}/aoi/user/{uid}"

        logger.info(f"Fetching AOIs for user: {uid}")

        try:
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Found {len(data) if isinstance(data, list) else 1} AOI(s)")
            return data if isinstance(data, list) else [data]

        except requests.HTTPError as e:
            logger.error(f"Failed to list AOIs: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def get_aoi(self, aoi_id: str) -> Dict:
        """
        Get details of a specific AOI by ID.

        Args:
            aoi_id: AOI identifier

        Returns:
            AOI details

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        url = f"{self.base_url}/aoi/{aoi_id}"

        logger.info(f"Fetching AOI: {aoi_id}")

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            logger.error(f"Failed to get AOI: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def delete_aoi(self, aoi_id: str) -> Dict:
        """
        Delete an AOI by ID.

        Args:
            aoi_id: AOI identifier

        Returns:
            Deletion response

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        url = f"{self.base_url}/aoi/delete/id/{aoi_id}"

        logger.info(f"Deleting AOI: {aoi_id}")

        try:
            response = self.session.delete(url)
            response.raise_for_status()

            logger.info(f"AOI {aoi_id} deleted successfully")
            return response.json()

        except requests.HTTPError as e:
            logger.error(f"Failed to delete AOI: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def download_all_products(
        self,
        aoi_id: str,
        user_id: Optional[str] = None,
    ) -> Dict:
        """
        Download all products for an AOI asynchronously (triggers email notification).

        Args:
            aoi_id: AOI identifier
            user_id: User ID (defaults to self.user_id)

        Returns:
            API response

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        uid = user_id or self.user_id
        url = f"{self.base_url}/download/all_products/{aoi_id}/{uid}"

        logger.info(f"Requesting all products for AOI {aoi_id}")

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Failed to download all products: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def download_max_flood_extent(
        self,
        aoi_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        output_type: str = "raster",
    ) -> Dict:
        """
        Submit order for maximum flood extent for an AOI and date range.

        Args:
            aoi_id: AOI identifier
            start_date: Start date (ISO format string or datetime)
            end_date: End date (ISO format string or datetime)
            output_type: Output type - one of: raster, vector, both

        Returns:
            Order response with order_id

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        # Convert dates to ISO format strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        # Use the correct endpoint from API docs
        url = f"{self.base_url}/download/max_flood_extent/{aoi_id}/{start_date}/{end_date}"

        params = {"output_type": output_type}

        logger.info(
            f"Submitting max flood extent order for AOI {aoi_id} from {start_date} to {end_date} ({output_type})"
        )

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Order submitted successfully")
            logger.info(f"Response: {data}")
            return data

        except requests.HTTPError as e:
            logger.error(f"Failed to download flood extent: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def _download_file(self, url: str, filepath: Path) -> None:
        """
        Download a file from URL to filepath.

        Args:
            url: URL to download from
            filepath: Local path to save file
        """
        response = self.session.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded: {filepath}")

    def _get_filename_from_response(
        self, response: requests.Response, default_prefix: str = "download"
    ) -> str:
        """
        Extract filename from response headers or generate default.

        Args:
            response: HTTP response
            default_prefix: Prefix for default filename

        Returns:
            Filename string
        """
        # Try to get filename from Content-Disposition header
        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[1].strip('"')
            return filename

        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{default_prefix}_{timestamp}.tif"

    def list_products(
        self,
        aoi_id: str,
        time: str = "all",
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> List[Dict]:
        """
        List available products for an AOI.

        Args:
            aoi_id: AOI identifier
            time: Time filter - one of: "latest", "range", "all" (default: "all")
            start_time: Start time for range query (YYYY-MM-DDTHH:MM:SS or datetime)
            end_time: End time for range query (YYYY-MM-DDTHH:MM:SS or datetime)

        Returns:
            List of product dictionaries

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        url = f"{self.base_url}/aoi/{aoi_id}/products"

        params = {"time": time}

        # Add time range if specified
        if time == "range":
            if start_time:
                if isinstance(start_time, datetime):
                    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
                params["from"] = start_time
            if end_time:
                if isinstance(end_time, datetime):
                    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
                params["to"] = end_time

        logger.info(f"Fetching products for AOI {aoi_id} (time={time})")

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            products = data if isinstance(data, list) else data.get("products", [])
            logger.info(f"Found {len(products)} product(s)")
            return products

        except requests.HTTPError as e:
            logger.error(f"Failed to list products: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def get_report_statistics(self, product_id: str) -> Dict:
        """
        Get report statistics for a product (includes affected population).

        Args:
            product_id: Product identifier

        Returns:
            Report statistics dictionary containing population impact data

        Raises:
            requests.HTTPError: If request fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        url = f"{self.base_url}/reporting/report_statistics/{product_id}"

        logger.info(f"Fetching report statistics for product {product_id}")

        try:
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            logger.info("Report statistics retrieved successfully")
            return data

        except requests.HTTPError as e:
            logger.error(f"Failed to get report statistics: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

    def download_product(
        self,
        product_id: str,
        download_dir: Union[str, Path] = "./data/gfm_rest",
    ) -> str:
        """
        Download entire product ensemble (all layers as zip).

        Args:
            product_id: Product identifier
            download_dir: Directory to save downloaded file

        Returns:
            Path to downloaded file

        Raises:
            requests.HTTPError: If download fails
        """
        if not self.token:
            logger.info("Not authenticated, logging in first")
            self.login()

        user_id = self.user_id
        url = f"{self.base_url}/download/product/{product_id}/{user_id}"

        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading product ensemble for {product_id}")

        try:
            response = self.session.get(url)
            response.raise_for_status()

            # Check if response contains a download link
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = response.json()
                download_link = data.get("download_link")

                if download_link:
                    logger.info(f"Following download link: {download_link}")

                    # Download from the provided link
                    download_response = self.session.get(download_link, stream=True)
                    download_response.raise_for_status()

                    # Get filename from download response
                    filename = self._get_filename_from_response(
                        download_response, default_prefix=f"{product_id[:8]}_ensemble"
                    )
                    filepath = download_dir / filename

                    # Download file
                    with open(filepath, "wb") as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    logger.info(f"Downloaded: {filepath}")
                    return str(filepath)
                else:
                    raise ValueError("No download_link in response")
            else:
                # Direct download
                filename = self._get_filename_from_response(
                    response, default_prefix=f"{product_id[:8]}_ensemble"
                )
                filepath = download_dir / filename

                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Downloaded: {filepath}")
                return str(filepath)

        except requests.HTTPError as e:
            logger.error(f"Failed to download product: {e}")
            logger.error(f"Response: {e.response.text}")
            raise


def main():
    """Example usage of GFMRestClient."""
    try:
        # Initialize client (credentials from environment)
        client = GFMRestClient()

        # Login
        login_info = client.login()
        print(f"Logged in as: {login_info.get('username')}")

        # List existing AOIs
        aois = client.list_aois()
        print(f"\nExisting AOIs: {len(aois)}")
        for aoi in aois:
            print(f"  - {aoi.get('name')} (ID: {aoi.get('aoi_id')})")

        # Example: Create a test AOI (small bbox in Cameroon)
        # bbox = [14.0, 9.0, 14.5, 9.5]  # [west, south, east, north]
        # aoi_response = client.create_aoi(
        #     name="Test_Cameroon_AOI",
        #     geometry=bbox,
        #     description="Test AOI for API testing"
        # )
        # print(f"\nCreated AOI: {aoi_response}")

    except Exception as e:
        logger.exception(f"Error: {e}")


if __name__ == "__main__":
    main()
