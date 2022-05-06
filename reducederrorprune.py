from sklearn import tree, metrics
from sklearn.model_selection import train_test_split

def ReducedErrorPrune(tree, node, parent, parentleft, x_test, y_test, clf, gparent, gparentleft):
    if node > -1:
        y_pred = clf.predict(x_test)
        preprune = metrics.mean_squared_error(y_test.astype(float), y_pred.astype(float))
        ReducedErrorPrune(tree, tree.children_left[node], node, 1, x_test, y_test, clf, parent, parentleft)
        ReducedErrorPrune(tree, tree.children_right[node], node, 0, x_test, y_test, clf, parent, parentleft)

        # print(tree.value)
        if ((tree.children_right[node] == -1) and (tree.children_left[node] == -1)):
            if (parentleft == 1) and (gparentleft == 1): 
                # print('children left left', tree.children_left[parent])
                tree.children_left[gparent] = tree.children_right[parent]
                
            elif (parentleft == 1) and (gparentleft == 0): 
                # print('children left right', tree.children_left[parent])
                tree.children_right[gparent] = tree.children_right[parent]

            elif (parentleft == 0) and (gparentleft == 0): 
                # print('children right right', tree.children_right[parent])
                tree.children_right[gparent] = tree.children_left[parent]
                
            elif (parentleft == 0) and (gparentleft == 1): 
                # print('children right left', tree.children_right[parent])
                tree.children_left[gparent] = tree.children_left[parent]

            # print('left tree', tree.children_left)
            # print('right tree', tree.children_right)
            y_pred = clf.predict(x_test)
            # print('HIT pred')
            postprune = metrics.mean_squared_error(y_test.astype(float), y_pred.astype(float))
            # print('postprune mmeansqr: ', postprune, node)
        
            if postprune > preprune: 
                if parentleft == 1: 
                    # print('restore left')
                    tree.children_left[parent] = node
                    # print('restore left after')
                else:
                    # print('restore right')
                    tree.children_right[parent] = node
                    # print('restore right after')


class DecisionTreeClassifierPruned(tree.DecisionTreeClassifier):
    def __init__(self, *,criterion="gini",splitter="best",max_depth=None,min_samples_split=2,min_samples_leaf=1,
                min_weight_fraction_leaf=0.,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.,
                min_impurity_split=None,class_weight=None,presort='deprecated',ccp_alpha=0.0):
        super().__init__(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,class_weight=class_weight,random_state=random_state,min_impurity_decrease=min_impurity_decrease,
                        min_impurity_split=min_impurity_split,presort=presort,ccp_alpha=ccp_alpha)

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        x_sub_train, x_sub_test, y_sub_train, y_sub_test = train_test_split(X, y)
        super().fit(
            x_sub_train, y_sub_train,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        ReducedErrorPrune(self.tree_, 0, -1, -1, x_sub_test, y_sub_test, self, -1, -1)
        return self


