package shell_structure;
import java.io.FileReader;
import java.lang.reflect.Method;
import com.google.gson.Gson;

public class Foo {
  public static void main(String[] args) throws Exception {
    Gson gson = new Gson();
    String request = gson.fromJson(new FileReader("input.json"));

    Class targetClass = Class.forName(request.dataClass);
    Object dataObject = gson.fromJson(request.data, targetClass);

    Method method = targetClass.getMethod(request.method);
    method.invoke(dataObject);
  }
}