//
//  main.cpp
//  剑指offer
//
//  Created by 蔡欣东 on 16/4/24.
//  Copyright © 2016年 蔡欣东. All rights reserved.
//

#include <iostream>
#include <vector>
#include <stack>
#include <unordered_map>
using namespace std;


struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x):val(x), next(NULL) {
        
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    
#pragma mark - 二维数组中的查找
    bool Find(vector<vector<int> > array,int target) {
        int n = 0;
        int m = (int)array[0].size()-1;
        while(n<array.size()&&m>=0){
            if(target>array[n][m]){
                n++;
            }else if(target<array[n][m]){
                m--;
            }else{
                return true;
            }
            
        }
        return false;
    }
    
#pragma mark - 替换空格
    void replaceSpace(char *str,int length) {
        int spaceCount = 0;
        for (int i=0; i<length; i++) {
            if (str[i]==' ') {
                spaceCount++;
            }
        }
        if (spaceCount==0) {
            return;
        }
        
        int newLen = length+2*spaceCount;
        char* index = str + length;
        while (index>=str) {
            if (*index==' ') {
                str[newLen--] = '0';
                str[newLen--] = '2';
                str[newLen--] = '%';
            }else{
                str[newLen--] = *index;
            }
            index--;
        }
    }
    
#pragma mark - 从尾到头打印链表
    vector<int> printListFromTailToHead(struct ListNode* head) {
        if (head==NULL) {
            return {};
        }else{
            stack<int> stack;
            vector<int> ve;
            ListNode* p = head;
            while (p!=NULL) {
                stack.push(p->val);
                p = p->next;
            }
            while (!stack.empty()) {
                int tmp = stack.top();
                ve.push_back(tmp);
                stack.pop();
            }
            return ve;
        }
    }
    
#pragma mark - 重建二叉树
    struct TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> in) {
        if(pre.size()<=0){
            return NULL;
        }else{
            TreeNode* root = new TreeNode(pre[0]);
            int n = (int)pre.size();
            vector<int> left_pre,left_in,right_pre,right_in;
            int p = 0;
            for(int i=0;i<n;i++){
                if(in[i]== pre[0]){
                    p = i;
                }
            }
            for(int i = 0;i<p;i++){
                left_pre.push_back(pre[i+1]);
                left_in.push_back(in[i]);
            }
            for(int i = p+1;i<n;i++){
                right_pre.push_back(pre[i]);
                right_in.push_back(in[i]);
            }
            //分治的思想
            root->left = reConstructBinaryTree(left_pre,left_in);
            root->right = reConstructBinaryTree(right_pre,right_in);
            return root;
        }
    }

#pragma mark - 旋转数组的最小数字
    int minNumberInRotateArray(vector<int> rotateArray) {
        if (rotateArray.size()==0) {
            return 0;
        }else{
            int first = 0;
            int last = (int)rotateArray.size()-1;
            int mid  = 0;
            while (rotateArray[first]>=rotateArray[last]) {
                mid = (first+last)/2;
                if (last-first==1) {
                    return rotateArray[last];
                }
                if (rotateArray[first]==rotateArray[mid]&&rotateArray[last]==rotateArray[mid]) {
                    int min = rotateArray[first];
                    for (int i = first+1; i<=last; i++) {
                        if (rotateArray[i]<min) {
                            min = rotateArray[i];
                        }
                    }
                    return min;
                }
                if (rotateArray[mid]>=rotateArray[first]) {
                    first = mid;
                }else{
                    last = mid;
                }
            }
            return rotateArray[first];
        }
    }
    
#pragma mark - 变态跳台阶问题
    int jumpFloorII(int number) {
        if (number<=0) {
            return -1;
        }else if (number==1){
            return 1;
        }else{
            return 1<<(number-1);
        }
    }
    
    

#pragma mark - 合并两个排序的链表,利用归并排序的思想
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2){
        if (pHead1==NULL) {
            return pHead2;
        }
        if (pHead2==NULL) {
            return pHead1;
        }
        ListNode* newHead = new ListNode(0);
        ListNode* go = newHead;
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        while (p1!=NULL&&p2!=NULL) {
            if (p1->val<p2->val) {
                go->next = p1;
                p1 = p1->next;
            }else{
                go->next = p2;
                p2 = p2->next;
            }
            go = go->next;
        }
        while (p1!=NULL) {
            go->next = p1;
            p1 = p1->next;
        }
        while (p2!=NULL) {
            go->next = p2;
            p2 = p2->next;
        }
        return newHead->next;
        
    }
    

#pragma mark - 调整数组顺序使奇数位于偶数前面,利用直接插入排序的思想
     void reOrderArray(vector<int> &array) {
        for (int i=1; i<array.size(); i++) {
            if (array[i]%2>0) {
                int j = i-1;
                int x = array[i];
                while (j>=0&&array[j]%2==0) {
                    array[j+1] = array[j];
                    j--;
                }
                array[j+1] = x;
            }
        }
    }
    

#pragma mark - 输入两颗二叉树A，B，判断B是不是A的子结构。
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2){
        if (pRoot1==NULL||pRoot2==NULL) {
            return false;
        }
        return isSubTree(pRoot1, pRoot2)||HasSubtree(pRoot1->left, pRoot2)||HasSubtree(pRoot1->right, pRoot2);
        
    }
    
    bool isSubTree(TreeNode* pRoot1,TreeNode* pRoot2){
        if (pRoot2 == NULL) return true;
        if (pRoot1 == NULL) return false;
        if (pRoot1->val==pRoot2->val) {
            return isSubTree(pRoot1->left, pRoot2->left)&&isSubTree(pRoot1->right, pRoot2->right);
        }else{
            return false;
        }
    }
    
#pragma mark - 二叉树的镜像
    void Mirror(TreeNode *pRoot) {
        if (pRoot==NULL) {
            return;
        }else{
            TreeNode* tmp = pRoot->left;
            pRoot->left = pRoot->right;
            pRoot->right = tmp;
            Mirror(pRoot->left);
            Mirror(pRoot->right);
        }
    }
    
#pragma mark - 顺时针打印矩阵
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> result;
        int row = (int)matrix.size();
        int col = (int)matrix[0].size();
        if (row==0||col==0) {
            return result;
        }
        int top = 0;
        int bottom = row-1;
        int left = 0;
        int right = col-1;
        while (top<=bottom&&left<=right) {
            for (int i = left; i<= right; i++) {
                result.push_back(matrix[top][i]);
            }
            for (int i = top+1; i<=bottom; i++) {
                result.push_back(matrix[i][right]);
            }
            if (top!=bottom) {
                for (int i = right-1; i>=left; i--) {
                    result.push_back(matrix[bottom][i]);
                }
            }
            if (left!=right) {
                for (int i = bottom-1; i>=top+1; i--) {
                    result.push_back(matrix[i][left]);
                }
            }
            top++;
            bottom--;
            left++;
            right--;
        }
        return  result;
    }
    

#pragma mark - 最长重复子串
    int lengthOfLongestSubstring(string s) {
        //保存每一个字母最近出现的位置
        int lastPos[256];
        memset(lastPos,-1,sizeof(lastPos));
        int maxLen = 0;
        int left = 0;
        int n = (int)s.length();
        for(int i = 0;i<n;i++){
            if(lastPos[s[i]]>=left){
                left = lastPos[s[i]]+1;
            }
            lastPos[s[i]] = i;
            maxLen = max(maxLen, i - left + 1);
        }
        return maxLen;
    }
    
#pragma mark - 最长回文字符串
    string longestPalindrome(string s) {
        
        if (s.length()<=1) {
            return s;
        }
        int n = (int)s.length();
        int max = 0;
        int maxLeft = 0;
        int maxRight = 0;
        for(int i=0;i<n;i++){
            //假定回文子串的长度为奇数
            int start = i-1;
            int end = i+1;
            int len = 1;
            int left = start;
            int right = end;
            while (start>=0&&end<n) {
                if (s[start]==s[end]) {
                    len = len+2;
                    left = start;
                    right = end;
                    start--;
                    end++;
                }else{
                    break;
                }
            }
            if (len>max) {
                max = len;
                maxLeft = left;
                maxRight = right;
            }
            //假定回文子串的长度为偶数
            start = i;
            end = i+1;
            len = 0;
            left = start;
            right = end;
            while (start>=0&&end<n) {
                if (s[start]==s[end]) {
                    len = len+2;
                    left = start;
                    right = end;
                    start--;
                    end++;
                }else{
                    break;
                }
            }
            if (len>max) {
                max = len;
                maxLeft = left;
                maxRight = right;
            }
        }
        return s.substr(maxLeft,maxRight-maxLeft+1);
    }
    
#pragma mark - 字符串转数字
    /**
     * 前面有空格
     * 不规格输出++，--
     * 前面有数字就输出到数字就可以了，例如 123+as 输出 123
     * 判断是否超过int的范围（-2^31~2^31-1）
     **/
    int myAtoi(string str) {
        int n = (int)str.length();
        long long num = 0;//注意
        int i = 0;
        int flag = 1;
        while (str[i]==' ') {
            i++;
        }
        if (str[i]=='-') {
            flag = -1;
            i++;
        }else if (str[i]=='+'){
            flag = 1;
            i++;
        }
        for (; i<n; i++) {
            if (str[i]>='0'&&str[i]<='9') {
                num = num*10+(str[i]-'0')*flag;
            }else{
                return (int)num;
            }
            if (num>2147483647) {
                return 2147483647;
            }else if (num<-2147483648){
                return -2147483648;
            }
            
        }

        return (int)num;
    }
    
#pragma mark - 数组中出现次数超过一半的数字
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        unordered_map<int, int> map;
        int n = (int)numbers.size();
        if (n==1) {
            return numbers[0];
        }
        for (int i = 0; i<n; i++) {
            if (map.count(numbers[i])>0) {
                int count = map[numbers[i]];
                count++;
                map[numbers[i]] = count;
                if (count>n/2) {
                    return numbers[i];
                }
            }else{
                map[numbers[i]] = 1;
            }
        }
        return 0;
    }
    
#pragma mark - 字符串全排列
    /**
     * 递归
     **/
    void swap(char &a,char &b){
        char tmp = a;
        a = b;
        b = tmp;
    }
    
    bool isToSwap(string str,int start,int end){
        for (int i = start; i<end; i++) {
            if (str[i]==str[end]) {
                return false;
            }
        }
        return true;
    }
    
    //k表示当前选取到第几个数,n表示共有多少数.
    void AllRange(vector<string> &ve, string str,int k,int n){
        if (k==n) {
            ve.push_back(str);
        }
        for (int i = k; i<n; i++) {
            if (isToSwap(str, k, i)) {
                swap(str[k], str[i]);
                AllRange(ve,str, k+1,n);
                swap(str[k], str[i]);
            }
        }
        
    }
    
    vector<string> Permutation(string str) {
        vector<string> ve;
        if (str.empty()) {
            return ve;
        }
        AllRange(ve, str, 0,(int)str.length());
        sort(ve.begin(), ve.end());
        return ve;
    }
#pragma mark - 连续子数组的最大和
    int FindGreatestSumOfSubArray(vector<int> array) {
        if (array.empty()) {
            return 0;
        }
        int mx = INT_MIN;
        int sum = 0;
        for (int i = 0; i<(int)array.size(); i++) {
            sum = sum + array[i];
            mx = max(mx, sum);
            if (sum<0) {
                sum = 0;
            }
        }
        return mx;
    }

#pragma mark - 翻转单词
    
    //利用vector
    string reverseWords(string str) {
        if (str.length()<=1) {
            return str;
        }
            vector<string> ve;
            size_t i = 0;
            size_t n = str.length();
            while (i<n && str[i]==' ') {
                i++;
            }
            while (n>0 && str[n-1]==' ' ) {
                n--;
            }
            size_t index = 0;
            while (index != -1 && index < n) {
                index = str.find_first_of(" ",i);
                string newStr = str.substr(i,index-i);
                ve.push_back(newStr);
                while (index < n && str[index] == ' ') {
                    index++;
                }
                i = index;
            }
    
            if (ve.empty()) {
                str = " ";
            }else{
                size_t size = ve.size();
                str = "";
                for (size_t i = size -1;i>0 ; i--) {
                    str = str + ve[i] + " ";
                }
                str = str + ve[0];
            }
        return str;
        
    }
    
};








int main(int argc, const char * argv[]) {
    Solution* s = new Solution();
    
    string str = " ";
    str = s->reverseWords(str);
    cout<<str<<endl;
        return 0;
}
