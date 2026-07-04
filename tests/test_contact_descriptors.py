import unittest

from ore_detection.descriptors.contacts import contact_lengths, hetero_sulfide_contact_length


class TestContactDescriptors(unittest.TestCase):
    def test_contact_length_counts_adjacent_class_edges_once(self):
        mask = [
            [1, 2],
            [1, 2],
        ]

        contacts = contact_lengths(mask, classes={1, 2})

        self.assertEqual(contacts[(1, 2)], 2)

    def test_hetero_sulfide_contact_ignores_background_contacts(self):
        mask = [
            [1, 2, 0],
            [1, 2, 0],
        ]

        length = hetero_sulfide_contact_length(mask, sulfide_classes={1, 2})

        self.assertEqual(length, 2)


if __name__ == "__main__":
    unittest.main()
