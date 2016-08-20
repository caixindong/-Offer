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
#include <queue>
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
    
#pragma mark - ====================查找算法====================
    void swapFunc(int &a, int &b){
        int temp = a;
        a = b;
        b = temp;
    }
    
#pragma mark - 直接插入排序
    void insertSort(int a[],int size){
        
        for (int i = 1; i < size; i ++) {
            if (a[i] < a[i-1]) {
                int j = i - 1;
                int x = a[i];
                a[i]  = a[j];
                while (x < a[j]) {
                    a[j+1] = a[j];
                    j--;
                }
                a[j+1] = x;
            }
        }
    }
    
#pragma mark - 希尔排序
    void normalInsertSort(int a[], int d, int size){
        for (int i = d; d < size; i ++) {
            if (a[i] < a[i - d]) {
                int j = i -d;
                int x = a[i];
                a[i]  = a[j];
                while (x < a[j]) {
                    a[j+d] = a[j];
                    j = j - d;
                }
                a[j+d] = x;
            }
        }
    }
    
    
    void shellSort(int a[],int size){
        int d = size/2;
        while (d >= 1) {
            normalInsertSort(a, d, size);
            d = d/2;
        }
    }
    
#pragma mark - 选择排序
    void selectSort(int a[], int size){
        for (int i = 0; i < size; i++) {
            int k = i;
            for (int j = i+1; j<size; j++) {
                if (a[j] < a[k]) {
                    k = j;
                }
            }
            swapFunc(a[i], a[k]);
        }
    }
    
#pragma mark - 冒泡排序
    void bubbleSort(int a[], int size){
        for (int i=0; i<size; i++) {
            for (int j = 0; j<size-i-1; j++) {
                if (a[j] > a[j+1]) {
                    swapFunc(a[j], a[j+1]);
                }
            }
        }
    }
    
#pragma mark - 堆排序
    void adjustHeap(int i, int size, int a[]){
        int left = 2*(i+1)-1;
        int right = 2*(i+1);
        int max = i;
        if (i <= size/2 - 1) {
            if (left < size && a[left] > a[max]) max = left;
            if (right < size && a[right] > a[max]) max = right;
            if (max != i) swapFunc(a[max], a[i]),adjustHeap(max, size, a);
        }
    }
    
    void setupHeap(int a[], int size){
        for (int i = size/2-1; i>=0; i--) {
            adjustHeap(i, size, a);
        }
    }
    
    void heapSort(int a[], int size) {
        setupHeap(a, size);
        for (int i = size-1 ; i > 0 ; i--) {
            swapFunc(a[0], a[i]);
            adjustHeap(0, size, a);
        }
    }
    
#pragma mark - 快速排序
    int partion(int start, int end, int a[]) {
        int i = start;
        int j = end;
        int x = a[i];
        while (i < j) {
            while (i < j && a[j] > x) j--;
            if (i < j) a[i] = a[j], i++;
            while (i < j && a[i] < x) i++;
            if (i < j) a[j] = a[i], j--;
        }
        a[i] = x;
        return i;
    }
    
    void subQuickSort(int start, int end, int a[]){
        if (start < end) {
            int p = partion(start, end, a);
            subQuickSort(start, p-1 , a);
            subQuickSort(p+1 , end, a);
        }
    }
    
    void quickSort(int a[], int size){
        subQuickSort(0, size-1, a);
    }
    
#pragma mark - 归并排序
    void mergeArray(int start,int mid, int end,int a[], int tmp[]){
        int i = start;
        int n = mid;
        int j = mid+1;
        int m = end;
        int k = 0;
        while (i <=n && j<=m) {
            if (a[i] < a[j]) {
                tmp[k++] = a[i++];
            }else {
                tmp[k++] = a[j++];
            }
        }
        while (i <= n) {
            tmp[k++] = a[i++];
        }
        while (j <= m) {
            tmp[k++] = a[j++];
        }
        for (int i = 0; i < k; i++) {
            a[start+i] = tmp[i];
        }
    }
    
    void subMergeSort(int start,int end ,int a[], int tmp[]){
        if (start < end) {
            int mid = (start+end)/2;
            subMergeSort(start, mid, a, tmp);
            subMergeSort(mid+1, end, a, tmp);
            mergeArray(start, mid, end, a, tmp);
        }
    }
    
    void mergeSort(int a[],int size){
        int *tmp = new int[size];
        subMergeSort(0, size-1, a, tmp);
        delete [] tmp;
    }
    
    
#pragma mark - ====================字符串====================
    
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
            str = "";
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
    
#pragma mark - 字符串全排列
    /**
     * 递归
     **/
    void swap(char &a,char &b){
        char tmp = a;
        a = b;
        b = tmp;
    }
    
    //判断当前交换位置的数字之前有没有出现，出现过就不交换
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
        //当前第k位的数跟它后面的数交换
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
    
#pragma mark - ====================数组====================
    
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
    
#pragma mark - 旋转数组的最小数字
    int minNumberInRotateArray(vector<int> rotateArray) {
        if (rotateArray.size()==0) {
            return 0;
        }else{
            int first = 0;
            int last = (int)rotateArray.size()-1;
            int mid  = 0;
            while (rotateArray[first]>=rotateArray[last]) {
                //终止条件
                if (last-first==1) {
                    return rotateArray[last];
                }
                mid = (first+last)/2;
                //如果中间的数比first的大说明最小值到后半段，否则在前半段
                if (rotateArray[mid]>=rotateArray[first]) {
                    first = mid;
                }else{
                    last = mid;
                }
                //无法判断大小的时候，只能顺序查找，如{1,0,1,1,1}
                if (rotateArray[first]==rotateArray[mid]&&rotateArray[last]==rotateArray[mid]) {
                    int min = rotateArray[first];
                    for (int i = first+1; i<=last; i++) {
                        if (rotateArray[i]<min) {
                            min = rotateArray[i];
                        }
                    }
                    return min;
                }
               
            }
            return rotateArray[first];
        }
    }

#pragma mark - 二分查找
    
    int binary_search(int start, int end, int key, int a[]) {
        int mid = 0;
        while (start <= end) {
            mid = (start + end)/2;
            if (a[mid] > key) {
                end = mid - 1;
            }else if (a[mid] < key){
                start = mid + 1;
            }else {
                return mid;
            }
        }
        return -1;
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
    
#pragma mark - 第 k 大的数
    void HeapAdjust(vector<int>& nums,int i, int size) {
        int left = 2*(i+1)-1;
        int right = 2*(i+1);
        int max = i;
        if (i <= size/2-1) {
            if (left < size && nums[left] > nums[max]) {
                max = left;
            }
            if (right < size && nums[right] > nums[max]) {
                max = right;
            }
            if (max != i) {
                swapFunc(nums[i], nums[max]);
                HeapAdjust(nums, max,size);
            }
        }
    }
    
    int findKthLargest(vector<int>& nums, int k) {
        int n = (int)nums.size();
        if (k > n) {
            return -1;
        }
        for (int i = n/2-1; i >= 0; i--) {
            HeapAdjust(nums, i, n);
        }
        for (int i = 1; i < k; i++) {
            swapFunc(nums[0], nums[n-i]);
            HeapAdjust(nums, 0 ,n-i);
        }
        return nums[0];
    }
    
#pragma mark - ====================链表====================
    
#pragma mark - 翻转链表
    ListNode* reverseList(ListNode* head) {
        if (head == NULL || head->next == NULL) {
            return head;
        }
        ListNode *pre = head;
        ListNode *p = head->next;
        ListNode *next = NULL;
        while (p != NULL) {
            next = p->next;
            p->next = pre;
            pre = p;
            p = next;
        }
        head->next = NULL;
        return pre;
    }
    
#pragma mark - 删除某个结点
    void deleteNode(ListNode* node) {
        //暂时包含后面那个结点
        ListNode *tmp = node->next;
        //node结点赋值下个结点的值以及next指针
        node->val = node->next->val;
        //删除后面那个结点
        node->next = node->next->next;
        delete tmp;
    }

#pragma mark - 删除倒数第n个结点
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if (head == NULL) {
            return head;
        }
        ListNode *newHead = new ListNode(0);
        newHead->next = head;
        ListNode *pre = newHead;
        ListNode *p = newHead;
        for (int i = 0; i < n; i++) {
            p = p->next;
        }
        while (p->next != NULL) {
            pre = pre->next;
            p = p->next;
        }
        pre->next = pre->next->next;
        return newHead->next;
    }
    
#pragma mark - 合并两个有序的链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == NULL) {
            return l2;
        }
        if (l2 == NULL) {
            return l1;
        }
        ListNode *newHead = new ListNode(0);
        ListNode *p = newHead;
        ListNode *p1 = l1;
        ListNode *p2 = l2;
        while (p1 != NULL && p2 != NULL ) {
            if (p1->val < p2->val) {
                p->next = p1;
                p1 = p1->next;
            }else {
                p->next = p2;
                p2 = p2->next;
            }
  
        
            p  = p ->next;
        }
        while (p1 != NULL) {
            p->next = p1;
            p1 = p1->next;
            p  = p->next;
        }
        while (p2 != NULL) {
            p->next = p2;
            p2 = p2->next;
            p  = p->next;
        }
        p->next = NULL;
        return newHead->next;
    }
    
#pragma mark - 判断链表是否有环
    bool hasCycle(ListNode *head) {
        if (head==NULL || head->next == NULL) {
            return false;
        }
        //定义一个快指针和慢指针
        ListNode *slow = head;
        ListNode *fast = head;
        while (fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
            //注意要判断fast不能为NULL，当快指针赶上慢指针的时候说明有环
            if (fast != NULL && (fast == slow || fast->next == slow)) {
                return true;
            }
        }
        return false;
    }
    
#pragma mark - 两个链表的第一个公共节点
    /**
     * 第一种情况 两个链表都没有环，那么他们是Y形相交，计算他们的长度，求出长度差k，分别给两个链表定义两个遍历的指针，长链表的指针先走k步，之后两个链表的指针同时走，一直到相同的结点；
     * 第二种情况，两个链表都有环，有两种交点，一种是进入环前有交点，另一种是链表入环的交点
     * 不可能一个有环一个无环，一个单链表最后一个结点的next是指向NULL，而有环的链表的结点的next不指向NULL，所以不可能有结点相交
     
     **/
    
#pragma mark - 删除链表中重复节点
    ListNode* deleteDuplicates(ListNode* head){
        if (head == NULL || head->next == NULL) {
            return head;
        }
        ListNode *pre = head;
        ListNode *p = pre->next;
        while (p != NULL) {
            if (pre->val == p->val) {
                pre->next = p->next;
                p = p->next;
            }else {
                pre = pre->next;
                p = p->next;
            }
        }
        return head;
    }
    
    
#pragma mark - ====================未分类====================
    
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


    
#pragma mark - 栈的压入和弹出序列
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if (pushV.size() ==0) {
            return false;
        }else{
            stack<int> stack;
            size_t size = pushV.size();
            size_t k = 0;
            for (size_t i = 0; i < size; i++) {
                if (!stack.empty() && stack.top() == popV[i]) {
                    stack.pop();
                }else{
                    while (true) {
                        if (k >= size) {
                            return false;
                        }else{
                            if (pushV[k] == popV[i]) {
                                k ++;
                                break;
                            }else{
                                stack.push(pushV[k]);
                                k ++;
                            }
                        }
                    }
                }
            }
            return true;

        }
    }
    
#pragma mark - 从上往下打印二叉树
    vector<int> PrintFromTopToBottom(TreeNode *root) {
        if (root == NULL) {
            return {};
        }
        vector<int> value = {};
        queue<TreeNode *> queue;
        queue.push(root);
        while (!queue.empty()) {
            TreeNode *node = queue.front();
            value.push_back(node->val);
            queue.pop();
            if (node->left != NULL) {
                queue.push(node->left);
            }
            if (node->right != NULL) {
                queue.push(node->right);
            }
        }
        return value;
    }
    
#pragma mark - 二叉搜索树的后序遍历序列
    bool VerifySquenceOfBST(vector<int> sequence) {
        if (sequence.size() <= 0) {
            return false;
        }
        size_t i = 0;
        size_t size = sequence.size();
        while (--size > 0) {
            //关键下面两步，左子树一定比根小，右子树一定比根大
            while(sequence[i++] < sequence[size]);
            while(sequence[i++] > sequence[size]);
            if (i < size) {
                return false;
            }
            i = 0;
        }
        return true;
    }
    
    
#pragma mark - 大数相乘
    void reverseString(string &str){
        size_t i = 0;
        size_t j = str.length()-1;
        for (; i < j; i++,j--) {
            char tmp = str[i];
            str[i] = str[j];
            str[j] = tmp;
        }
    }
    
    
    string multiLargeNum(string a, string b){
        size_t n = a.length();
        size_t m = b.length();
        string result(n + m - 1, '0');
        //逆序更符合我们的运算习惯
        reverseString(a);
        reverseString(b);
        //乘法进位
        int multFlag;
        //加法进位
        int addFlag;
        for (size_t i = 0; i < n ; i++) {
            multFlag = 0;
            addFlag = 0;
            for (size_t j = 0; j < m; j++) {
                int temp1 = (a[i]-'0')*(b[j]-'0')+multFlag;
                multFlag = temp1/10;
                temp1 = temp1%10;
                int temp2 = (result[i+j]-'0')+temp1+addFlag;
                addFlag = temp2/10;
                result[i+j] = '0'+temp2%10;
            }
            result[i+m] += multFlag + addFlag;
        }
        reverseString(result);
        return result;
    }
    
};








int main(int argc, const char * argv[]) {
    Solution* s = new Solution();
    ListNode *l1 = new ListNode(1);
    ListNode *l2 = new ListNode(2);
    l1->next = l2;
    cout<<s->hasCycle(l1)<<endl;
    return 0;
}
