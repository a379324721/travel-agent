"""Travel inventory search: flights, hotels, trains."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TravelMode(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    TRAIN = "train"


class FlightSearchRequest(BaseModel):
    origin: str = Field(..., description="IATA 或城市")
    destination: str
    depart_date: date
    return_date: date | None = None
    cabin: str = "economy"
    passengers: int = 1


class HotelSearchRequest(BaseModel):
    city: str
    check_in: date
    check_out: date
    keyword: str | None = None


class TrainSearchRequest(BaseModel):
    origin_station: str
    dest_station: str
    depart_date: date
    prefer_gd: bool = True


async def search_flights(req: FlightSearchRequest) -> dict[str, Any]:
    return {
        "mode": TravelMode.FLIGHT.value,
        "query": req.model_dump(mode="json"),
        "results": [
            {
                "carrier": "CA",
                "flight_no": "CA1501",
                "dep": f"{req.origin} 08:00",
                "arr": f"{req.destination} 10:30",
                "price_cny": 1280,
            }
        ],
    }


async def search_hotels(req: HotelSearchRequest) -> dict[str, Any]:
    nights = max((req.check_out - req.check_in).days, 1)
    return {
        "mode": TravelMode.HOTEL.value,
        "query": req.model_dump(mode="json"),
        "results": [
            {"name": f"{req.city}商务酒店", "star": 4, "nightly_cny": 560, "nights": nights}
        ],
    }


async def search_trains(req: TrainSearchRequest) -> dict[str, Any]:
    return {
        "mode": TravelMode.TRAIN.value,
        "query": req.model_dump(mode="json"),
        "results": [
            {
                "train_no": "G103" if req.prefer_gd else "K101",
                "dep": f"{req.origin_station} 09:20",
                "arr": f"{req.dest_station} 14:05",
                "seat": "二等座",
                "price_cny": 553,
            }
        ],
    }
