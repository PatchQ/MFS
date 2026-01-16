import requests
from bs4 import BeautifulSoup
import re

class WebsiteLoginScraper:
    def __init__(self, base_url=None):
        """
        初始化爬蟲
        
        Args:
            base_url (str): 網站基礎URL
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = base_url
        self.logged_in = False
        
    def login(self, login_url, username, password, login_data=None, method='post'):
        """
        登錄網站
        
        Args:
            login_url (str): 登錄頁面URL
            username (str): 用戶名
            password (str): 密碼
            login_data (dict): 登錄表單數據，如果為None則自動構建
            method (str): 請求方法，'post'或'get'
            
        Returns:
            bool: 登錄是否成功
        """
        try:
            # 如果需要，先獲取登錄頁面以獲取CSRF令牌等
            if login_data is None:
                # 嘗試獲取登錄頁面來分析表單
                try:
                    response = self.session.get(login_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 尋找常見的登錄表單字段
                    login_data = {
                        'username': username,
                        'password': password,
                        'email': username  # 有些網站使用email字段
                    }
                    
                    # 嘗試尋找CSRF令牌
                    csrf_token = self._find_csrf_token(soup)
                    if csrf_token:
                        login_data[csrf_token['name']] = csrf_token['value']
                        
                except Exception as e:
                    print(f"獲取登錄頁面時出錯: {e}")
                    # 如果無法獲取頁面，使用基本的登錄數據
                    login_data = {
                        'username': username,
                        'password': password
                    }
            
            # 發送登錄請求
            if method.lower() == 'post':
                response = self.session.post(login_url, data=login_data)
            else:
                response = self.session.get(login_url, params=login_data)
            
            # 檢查登錄是否成功
            self.logged_in = self._check_login_success(response)
            
            if self.logged_in:
                print("登錄成功!")
                return True
            else:
                print("登錄失敗，請檢查用戶名、密碼或登錄數據")
                return False
                
        except Exception as e:
            print(f"登錄過程中出錯: {e}")
            return False
    
    def _find_csrf_token(self, soup):
        """
        在HTML中查找CSRF令牌
        
        Args:
            soup (BeautifulSoup): BeautifulSoup對象
            
        Returns:
            dict or None: 包含name和value的字典，如果找不到則返回None
        """
        # 查找常見的CSRF令牌字段
        csrf_patterns = ['csrf', 'token', 'authenticity', '_token', 'csrfmiddlewaretoken']
        
        # 在input標籤中查找
        for input_tag in soup.find_all('input'):
            input_name = input_tag.get('name', '').lower()
            input_value = input_tag.get('value', '')
            
            for pattern in csrf_patterns:
                if pattern in input_name and input_value:
                    return {'name': input_tag.get('name'), 'value': input_value}
        
        # 在meta標籤中查找
        for meta_tag in soup.find_all('meta'):
            meta_name = meta_tag.get('name', '').lower()
            meta_content = meta_tag.get('content', '')
            
            for pattern in csrf_patterns:
                if pattern in meta_name and meta_content:
                    return {'name': meta_tag.get('name'), 'value': meta_content}
        
        return None
    
    def _check_login_success(self, response):
        """
        檢查登錄是否成功
        
        Args:
            response (requests.Response): 登錄請求的響應
            
        Returns:
            bool: 登錄是否成功
        """
        # 檢查狀態碼
        if response.status_code != 200:
            return False
        
        # 檢查響應內容中是否包含登錄成功或失敗的提示
        success_indicators = ['登出', 'logout', '我的帳戶', 'my account', 'welcome', 'dashboard']
        failure_indicators = ['登錄失敗', 'login failed', 'invalid', '錯誤', 'incorrect']
        
        text_lower = response.text.lower()
        
        # 如果有失敗指示器，返回False
        for indicator in failure_indicators:
            if indicator in text_lower:
                return False
        
        # 如果有成功指示器，返回True
        for indicator in success_indicators:
            if indicator in text_lower:
                return True
        
        # 如果沒有明確的指示器，檢查是否有常見的登錄後元素
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找常見的登錄後元素
        logged_in_elements = soup.find_all(['a', 'div', 'span'], 
                                           text=re.compile(r'登出|logout|我的帳戶|my account', re.IGNORECASE))
        
        if logged_in_elements:
            return True
        
        # 如果都不確定，檢查cookie中是否有會話信息
        if 'session' in str(self.session.cookies).lower() or 'auth' in str(self.session.cookies).lower():
            return True
        
        return False
    
    def fetch_page(self, url, params=None):
        """
        獲取頁面內容
        
        Args:
            url (str): 要訪問的URL
            params (dict): 查詢參數
            
        Returns:
            BeautifulSoup: 解析後的HTML對象
        """
        try:
            if not self.logged_in:
                print("請先登錄")
                return None
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                return BeautifulSoup(response.text, 'html.parser')
            else:
                print(f"獲取頁面失敗，狀態碼: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"獲取頁面時出錯: {e}")
            return None
    
    def extract_data(self, soup, selectors):
        """
        從HTML中提取數據
        
        Args:
            soup (BeautifulSoup): BeautifulSoup對象
            selectors (dict): CSS選擇器字典，格式為{'key': 'selector'}
            
        Returns:
            dict: 提取的數據
        """
        data = {}
        
        if not soup:
            return data
        
        for key, selector in selectors.items():
            try:
                elements = soup.select(selector)
                if elements:
                    if len(elements) == 1:
                        # 如果只有一個元素，提取文本或屬性
                        if elements[0].name in ['img', 'a', 'link']:
                            # 對於特定標籤，提取常用屬性
                            if elements[0].has_attr('src'):
                                data[key] = elements[0]['src']
                            elif elements[0].has_attr('href'):
                                data[key] = elements[0]['href']
                            else:
                                data[key] = elements[0].get_text(strip=True)
                        else:
                            data[key] = elements[0].get_text(strip=True)
                    else:
                        # 如果有多個元素，提取所有文本
                        data[key] = [elem.get_text(strip=True) for elem in elements]
                else:
                    data[key] = None
            except Exception as e:
                print(f"提取數據時出錯 ({key}): {e}")
                data[key] = None
        
        return data
    
    def find_all_links(self, soup, pattern=None):
        """
        查找頁面中的所有鏈接
        
        Args:
            soup (BeautifulSoup): BeautifulSoup對象
            pattern (str): 正則表達式模式，用於過濾鏈接
            
        Returns:
            list: 鏈接列表
        """
        if not soup:
            return []
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if pattern:
                if re.search(pattern, href):
                    links.append({
                        'text': a_tag.get_text(strip=True),
                        'url': href,
                        'full_url': self._make_full_url(href)
                    })
            else:
                links.append({
                    'text': a_tag.get_text(strip=True),
                    'url': href,
                    'full_url': self._make_full_url(href)
                })
        
        return links
    
    def _make_full_url(self, url):
        """
        將相對URL轉換為絕對URL
        
        Args:
            url (str): 相對或絕對URL
            
        Returns:
            str: 完整URL
        """
        if url.startswith('http://') or url.startswith('https://'):
            return url
        elif self.base_url:
            if url.startswith('/'):
                # 從基礎URL構建完整URL
                base = self.base_url.rstrip('/')
                return f"{base}{url}"
            else:
                return f"{self.base_url}/{url}"
        else:
            return url
    
    def logout(self):
        """登出"""
        self.logged_in = False
        self.session.close()
        print("已登出")
    
    def __del__(self):
        """析構函數，確保會話關閉"""
        self.session.close()


# 示例使用 - 假設的網站登錄
def example_usage():
    # 創建爬蟲實例
    scraper = WebsiteLoginScraper(base_url="https://portal.iata.org/s/login/?language=en_US")
    
    # 登錄信息
    login_url = "https://portal.iata.org/s/login/?language=en_US"
    username = "chanisabel@dsi.gov.mo"
    password = "Dsimsar529."
    
    # 登錄數據（根據實際網站調整）
    login_data = {
        'username': username,
        'password': password,
        'remember': 'on'
    }
    
    # 執行登錄
    if scraper.login(login_url, username, password, login_data):
        # 登錄成功後訪問受保護的頁面
        protected_page_url = "https://timaticweb.iata.org/#/portal"
        soup = scraper.fetch_page(protected_page_url)
        
        if soup:
            # 定義要提取的數據的CSS選擇器
            selectors = {
                'user_name': '.user-name',
                'user_email': '.user-email',
                'dashboard_title': 'h1.dashboard-title',
                'stats': '.stat-item',
                'recent_activity': '.activity-list li'
            }
            
            # 提取數據
            data = scraper.extract_data(soup, selectors)
            
            print("提取的數據:")
            for key, value in data.items():
                print(f"{key}: {value}")
            
            # 查找所有鏈接
            print("\n頁面鏈接:")
            links = scraper.find_all_links(soup, pattern=r'dashboard|profile|settings')
            for link in links[:5]:  # 只顯示前5個鏈接
                print(f"{link['text']}: {link['full_url']}")
        
        # 登出
        scraper.logout()
    else:
        print("登錄失敗，無法繼續")



if __name__ == "__main__":
    print("=" * 50)
    print("BeautifulSoup登錄後數據分析示例")
    print("=" * 50)
    
    # 運行示例
    example_usage()
    