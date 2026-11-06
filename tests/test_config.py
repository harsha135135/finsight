import unittest

from aiagents.config import Job


class ConfigTests(unittest.TestCase):
    def valid(self, **changes):
        return {"goal": "Research", "sources": [{"id": "evidence", "path": "data.md"}], **changes}

    def test_rejects_unknown_fields_and_boolean_budget(self):
        for job in (self.valid(execute_shell=True), self.valid(max_model_calls=True), self.valid(min_sources=2)):
            with self.assertRaises(ValueError):
                Job.from_dict(job)

    def test_rejects_ambiguous_sources_and_case_collisions(self):
        for sources in (
            [{"id": "x", "path": "a.md", "url": "https://example.com"}],
            [{"id": "X", "path": "a.md"}, {"id": "x", "path": "b.md"}],
            [{"id": "<script>", "path": "a.md"}],
        ):
            with self.assertRaises(ValueError):
                Job.from_dict(self.valid(sources=sources))

    def test_round_trip(self):
        job = Job.from_dict(self.valid())
        self.assertEqual(Job.from_dict(job.to_dict()), job)


if __name__ == "__main__":
    unittest.main()
