package br.com.slmlab.avaliador.config;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
@Configuration
public class BrowserFreshnessConfiguration implements WebMvcConfigurer {
    @Override public void addInterceptors(InterceptorRegistry registry){
        registry.addInterceptor(new HandlerInterceptor(){
            @Override public boolean preHandle(HttpServletRequest request,HttpServletResponse response,Object handler){
                response.setHeader("Cache-Control","no-store");return true;
            }
        }).addPathPatterns("/api/**","/","/labs/**","/docs/**","/datasets");
    }
}
