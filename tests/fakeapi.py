import csv
import io
import json
import urllib.parse


class FakeApi:
    def __init__(self, financials, failures, fed_files):
        self.financials = financials
        self.failures = failures
        self.fed_files = fed_files
        self.calls = 0
        self.fail_first = 0

    def _rows(self, endpoint, filters):
        if endpoint == "failures":
            return self.failures
        date = filters.split(":", 1)[1]
        return [row for row in self.financials if row["REPDTE"] == date]

    def __call__(self, request, timeout):
        self.calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise OSError("connection reset")
        url = urllib.parse.urlsplit(request.full_url)
        if url.netloc == "fed.example":
            name = url.path.rsplit("/", 1)[-1]
            if name not in self.fed_files:
                raise OSError("404")
            return self.fed_files[name]
        endpoint = url.path.rsplit("/", 1)[-1]
        query = dict(urllib.parse.parse_qsl(url.query))
        rows = self._rows(endpoint, query.get("filters", ""))
        if query.get("format") != "csv":
            return json.dumps({"meta": {"total": len(rows)}, "data": []}).encode()
        key = query["sort_by"]
        ordered = sorted(rows, key=lambda row: str(row[key]).zfill(12))
        offset = int(query.get("offset", 0))
        page = ordered[offset : offset + int(query["limit"])]
        fields = sorted(set(query["fields"].split(",")) | {"ID"})
        buffer = io.StringIO()
        writer = csv.writer(buffer, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        writer.writerow(fields)
        for row in page:
            writer.writerow([row.get(field, "") for field in fields])
        return buffer.getvalue().encode()
