import unittest

from ore_detection.data.ore_type_legend import load_legend_config


class TestOreTypeLegend(unittest.TestCase):
    def setUp(self):
        self.legend = load_legend_config()

    def test_background_is_explicit_channel(self):
        self.assertEqual(self.legend.class_names[self.legend.background_index], "background")
        self.assertNotIn(self.legend.background_index, self.legend.non_background_indices)

    def test_same_full_name_merges_to_same_target_index(self):
        set1_chalcopyrite = self.legend.color_to_entry("set_1")[(255, 165, 0)]
        set2_chalcopyrite = self.legend.color_to_entry("set_2")[(255, 165, 0)]
        set3_chalcopyrite = self.legend.color_to_entry("set_3")[(255, 165, 0)]

        self.assertEqual(set1_chalcopyrite.target, "chalcopyrite")
        self.assertEqual(set1_chalcopyrite.class_index, set2_chalcopyrite.class_index)
        self.assertEqual(set1_chalcopyrite.class_index, set3_chalcopyrite.class_index)

    def test_rare_set3_legend_classes_are_mapped(self):
        entries = self.legend.color_to_entry("set_3")

        self.assertEqual(entries[(0, 0, 139)].target, "marcasite")
        self.assertEqual(entries[(139, 0, 139)].target, "native_gold")
        self.assertEqual(entries[(0, 128, 0)].target, "covellite")

    def test_talc_is_dummy_output_not_supervised_class(self):
        self.assertNotIn("talc", self.legend.class_names)
        self.assertEqual(self.legend.dummy_outputs[0]["name"], "talc")
        self.assertFalse(self.legend.dummy_outputs[0]["trained"])


if __name__ == "__main__":
    unittest.main()
