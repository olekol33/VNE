from include import config
from functools import lru_cache

class ElementTranslator:
    @staticmethod
    def _make_node(i, m):
        return i, m

    @staticmethod
    def _make_link(i, j, m):
        return i, j, m

    @staticmethod
    def _unpack_element(element):
        return element

    @staticmethod
    def get_node_to_node_edge(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        assert m == n
        return ElementTranslator._make_node(i, m), ElementTranslator._make_node(j, n)

    @staticmethod
    def get_node_to_self_node_edge(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        assert m == n
        return ElementTranslator._make_link(i, j, m), ElementTranslator._make_node(j, n)

    @staticmethod
    def get_link_to_node_edge(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        assert m != n
        return ElementTranslator._make_link(i, j, m), ElementTranslator._make_node(j, n)

    @staticmethod
    def get_link_to_self_node_edge(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        return ElementTranslator._make_link(i, j, n), ElementTranslator._make_node(j, n)

    @staticmethod
    def get_link(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        return ElementTranslator._make_link(i, j, n)
        # return ElementTranslator._make_node(i, m), ElementTranslator._make_node(j, n)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_node(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        return ElementTranslator._make_node(i, m)

    @staticmethod
    def get_node_to_link_edge(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        return ElementTranslator._make_node(i, m), ElementTranslator._make_link(i, j, n)

    @staticmethod
    def get_link_to_link_edge(element):
        i, j, m, n = ElementTranslator._unpack_element(element)
        assert m != n
        return ElementTranslator._make_link(i, j, m), ElementTranslator._make_link(i, j, n)

    @staticmethod
    def phys_g_elem_to_alloc_g_mult(graph, elements: []) -> float:
        if len(elements) == 1:
            i, j, m, n = elements[0]
        else:
            i = j = m = n = None
            for element in reversed(elements):
                i, j, m, n = element
                if m == n:
                    break
        return graph.multiplier[graph.app.get_app_name(), i, j, m, n]


    @staticmethod
    @lru_cache(maxsize=10000)
    def alloc_graph_to_physical_graph_elements(graph, link: ()) -> ():
        """Convert from allocation graph links to element of type [i, j, m, n]"""
        elements = []
        from_node = link[0]
        to_node = link[1]
        from_node_type = graph.get_element_type(from_node)
        to_node_type = graph.get_element_type(to_node)

        if graph.element_is_allocated_node(from_node_type):
            if graph.element_is_allocated_node(to_node_type): #node-node
                i, m = from_node
                j, n = to_node
                if m == n:
                    elements.append((i, j, m, m))

                else:
                    elements.append((i, j, m, n))
                    elements.append((i, j, n, n))
            elif graph.element_is_agg_node(to_node_type): #node-agg
                pass
                # i, m = from_node
                # j = config.agg_node
                # n = to_node[1]
                # elements.append((i, j, m, m))
            else: #node-link
                i, m = from_node
                _, j, n = to_node
                elements.append((i, j, m, n))
                assert m != n
        elif graph.element_is_agg_node(from_node_type):
            if graph.element_is_allocated_node(to_node_type):
                i, m = from_node[0], from_node[1]
                j, n = to_node
                elements.append((i, j, m, n))
            elif graph.element_is_agg_node(to_node_type):
                raise ValueError("agg-node to agg-node not supported")
            else:
                i, m = from_node[0], from_node[1]
                _, j, n = to_node
                elements.append((i, j, m, n))
                assert m != n
        else:
            if graph.element_is_allocated_node(to_node_type): #link-node
                i, j1, m = from_node
                j2, n = to_node
                elements.append((i, j1, m, n))
                # elements.append((i, j1, n, n))
                assert m == n
                assert j1 == j2
            elif graph.element_is_agg_node(to_node_type): #link-agg
                raise ValueError("link-agg not supported")
            else: #link-link
                i1, j1, m = from_node
                i2, j2, n = to_node
                elements.append((i1, j1, m, n))
                assert m != n
                assert i1 == i2
                assert j1 == j2
        return elements

    @staticmethod
    # @lru_cache(maxsize=100000)
    def edge_to_multiplier_element(graph, model_element) -> ():
        # i, j, m, n = ElementTranslator.edge_to_model_element(graph, link)
        i, j, m, n = model_element
        if i == config.agg_node or j == config.agg_node:
            return graph.req.app_name, i, j, m, n
        elif m == n:
            return graph.req.app_name, j, j, m, m
        else:
            return graph.req.app_name, i, j, m, n
