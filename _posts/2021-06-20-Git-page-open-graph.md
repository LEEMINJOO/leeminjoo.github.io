---
layout: post
toc: true
title: "Jekyll Open Graph 설정하기"
categories: "Jekyll-setting"
---

[Jekyll 링크 미리보기 추가하기]({% post_url 2021-06-20-Git-page-link-preview %})에서 포스트 미리보기를 설정했습니다.

설정 과정에서 제 블로그에 Thumbnail 이미지가 없는 것을 확인해, 추가하는 과정에 대해 포스팅을 작성하였습니다.

## Thumbnail 미리보기

Thumbnail이 없을 경우 미리보기 결과 아래처럼 보입니다.

<figure>
    <center>
        <img src="/assets/imgs/jekyll/fail.png" 
         width="90%" height="90%" alt=""/> 
        <!-- <figcaption></figcaption> -->
    </center>
</figure>

(하지만 이때 KakaoTalk 미리보기에서는 Logo가 잘 보이고 있었습니다.)

<figure>
    <img src="/assets/imgs/jekyll/kakao-success.png" 
        width="90%" height="90%" alt="" align="middle"/> 
    <!-- <figcaption></figcaption> -->
</figure>

### metatags.io

Jekyll에 미리보기를 설정하기 전에 확인하는 페이지를 소개해 드리겠습니다.
구글과 슬랙, 페이스북에서 보여지는 화면을 미리 볼 수 있는 [metatags.io]("https://metatags.io/")입니다.

[metatags.io]("https://metatags.io/") 내에서 링크 검색 시 Thubmnail 이미지가 없는 것을 확인 할 수 있습니다.

<figure>
    <center>
        <img src="/assets/imgs/jekyll/metatag-fail.png" 
         width="90%" height="90%" alt="" /> 
        <figcaption>metatags.io</figcaption>
    </center>
</figure>

## Thumbnail 설정하기

metatags.io 오른쪽 상단에 `Generate meta tags` 클릭 시 
Open Graph를 설정할 수 있는 html 코드를 확인 할 수 있습니다.

<figure>
    <center>
        <img src="/assets/imgs/jekyll/gen-meta-tags.png" 
         width="90%" height="90%" alt=""/> 
        <figcaption>Generate meta tags</figcaption>
    </center>
</figure>

코드를 복사해  `default.html` 내부 <head> 아래 붙여넣기 합니다.
`property="og:image"`인 meta 정보의 `content`를 원하는 이미지의 주로소 변경합니다.

```html
<!-- Primary Meta Tags -->
<title>LEEMINJOO | Data Science</title>
<meta name="title" content="LEEMINJOO | Data Science">
<meta name="description" content="Data Science">

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://leeminjoo.github.io/">
<meta property="og:title" content="LEEMINJOO | Data Science">
<meta property="og:description" content="Data Science">
<meta property="og:image" content="https://leeminjoo.github.io/logo.png">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="https://leeminjoo.github.io/">
<meta property="twitter:title" content="LEEMINJOO | Data Science">
<meta property="twitter:description" content="Data Science">
<meta property="twitter:image" content="https://leeminjoo.github.io/logo.png">
```

## 결과

이제 아래와 같이 Thumnail 이미지가 표시됩니다.

<figure>
    <center>
        <img src="/assets/imgs/jekyll/success.png" 
         width="90%" height="90%" alt=""/> 
        <!-- <figcaption></figcaption> -->
    </center>
</figure>

metatags.io 에서도 아래와 같이 확인됩니다.

<figure>
    <center>
        <img src="/assets/imgs/jekyll/metatag-success.png" 
         width="90%" height="90%" alt="" /> 
        <figcaption>metatags.io</figcaption>
    </center>
</figure>
