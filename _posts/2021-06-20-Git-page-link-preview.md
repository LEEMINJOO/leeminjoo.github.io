---
layout: post
title: "Jekyll 링크 미리보기 추가하기"
categories: "Jekyll-setting"
---

Jekyll 기반 Gihub Page에 아래와 같이 링크 미리보기 설정 방법을 정리했습니다.

{% raw %}
{% linkpreview "https://leeminjoo.github.io" %}
{% endraw %}

## jekyll-linkpreview

아래 페이지를 참고하였습니다.

{% raw %}
{% linkpreview "https://github.com/ysk24ok/jekyll-linkpreview" %}
{% endraw %}

### 1. `_config.yml`

`_config.yml` 내에서 `plugins`에 `jekyll-linkpreview`를 추가합니다.

```yaml
plugins:
    ......
  - jekyll-linkpreview
```

### 2. `Gemfile`

`Gemfile` 내에 아래 코드를 추가합니다.

```
group :jekyll_plugins do
    gem 'jekyll-sitemap'
    gem 'jekyll-feed'
    gem 'jekyll-seo-tag'
    gem 'jekyll-linkpreview'
  end 
```

### 3. `linkpreview.css`

[`linkpreview.css`](https://github.com/ysk24ok/jekyll-linkpreview/blob/master/assets/css/linkpreview.css) 파일을 `_includes/` 폴더 아래 추가합니다.

저는 개인적으로 약간 수정해 아래 내용을 추가했습니다.

```css
  .jekyll-linkpreview-wrapper {
    max-width: 1200px;
    margin-top: 10px;
    margin-bottom: 10px;
  }

  .jekyll-linkpreview-wrapper-inner {
    border: 1px solid rgba(0,0,0,.1);
    padding: 12px;
  }

  .jekyll-linkpreview-content {
    position: relative;
    height: 120px;
    /* overflow: hidden; */
    margin-top: 5px;
    margin-bottom: 10px;
  }

  .jekyll-linkpreview-image {
    position: absolute;
    top: 0;
    right: 0;
  }

  .jekyll-linkpreview-image img {
    width: 130px;
    height: 130px;
  }

  .jekyll-linkpreview-body {
    margin-top: 10px;
    margin-right: 200px;
  }

  h2.jekyll-linkpreview-title{
    font-size: 20px;
    margin-top: 2px;
    margin: 0 0 2px;
    line-height: 1.3;
    /* display:block; Add this  */
  }

  .jekyll-linkpreview-description {
    line-height: 1.5;
    font-size: 13px;
    margin-top: 10px;
  }

  .jekyll-linkpreview-footer {
    font-size: 0px;
  } 
```

`linkpreview.css`를 읽어드리도록 `'_includes/head.html` 내에 아래 코드를 추가합니다.

```html
  <!-- Customized css -->
  <link rel="stylesheet" type="text/css" href="/assets/css/linkpreview.css" media="screen">
```

코드의 위치는 [전체 코드](https://github.com/LEEMINJOO/leeminjoo.github.io/blob/master/_includes/head.html) 참고 부탁드립니다.

### 4. Markdown 문법

포스트 작성 시 아래와 같이 `linkpreview` 를 통해 미리보기를 추가할 수 있습니다.
(아래 코드에서 `\` 삭제해 사용해야합니다.)


> ```\{ % linkpreview "https://leeminjoo.github.io" % \}```


## 결과

위 과정을 통해 미리보기 링크를 설정할 수 있습니다.

제 블로그는 아래와 같이 미리보기 이미지가 없어 추가로 설정하는 과정이 필요했습니다.
해당 내용은 다음 포스트 ["Jekyll Open Graph 설정하기"]({% post_url 2021-06-20-Git-page-open-graph %})참고 부탁드립니다.

<figure>
    <center>
        <img src="/assets/imgs/jekyll/fail.png" 
         width="90%" height="90%" alt=""/> 
        <!-- <figcaption></figcaption> -->
    </center>
</figure>

